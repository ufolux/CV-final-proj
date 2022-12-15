import os
import time

import ray
import collections
import numpy as np
import torch
from policies import A2C
import spiral.environments.libmypaint as libmypaint
import torch.nn.functional as F
from torchvision.utils import save_image
from config import SpiralConfig
from storage import SharedStorage

@ray.remote
class Demo_Painter():
    def __init__(self, config, label, seed=7):
        self.config = config
        self.env = self._init_env()
        self._action_spec = config.action_spec
        np.random.seed(seed)

        # init policy on cpu
        self.n = 0
        self.root = os.getcwd()
        self.a2c = A2C(self.env.action_spec(), input_shape=config.input_shape, grid_shape=config.grid_shape,
                       action_order="libmypaint")
        # self.a2c.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.a2c.to('cpu')
        self.a2c.eval()
        self.steps = 0
        self.label = torch.tensor([label])
        os.makedirs(self.root+f'/demo/{label}')
        self.logdir = self.root+f'/demo/{label}/sample_'
        self._reset_trajectory()


    def _init_env(self):
        env = libmypaint.LibMyPaint(**self.config.libmypaint_params)
        return env

    def _reset_trajectory(self):
        self.time_step = self.env.reset()
        self.noise_sample_no_label = torch.randn(1, 10)

        self.label_onehot = F.one_hot(self.label % 5, num_classes=5).float()
        self.noise_sample = torch.cat((self.noise_sample_no_label, self.label_onehot), 1)
        self.state = self.a2c.initial_state(self.label_onehot)
        self.steps = 0
        self.trajectory = {
            'images': [],
            'action_masks': collections.OrderedDict(
                [(spec, [self.time_step.observation['action_mask'][spec]]) for spec in self._action_spec]),
            'prev_actions': collections.OrderedDict(
                [(spec, [self.state.prev_action[spec]]) for spec in self._action_spec]),
            'actions': collections.OrderedDict([(spec, []) for spec in self._action_spec]),
            'values': [],
            'noise_sample': self.noise_sample.squeeze().numpy(),
        }
        self.log_dir_child = self.logdir + str(self.n) + '/'
        if not os.path.exists(self.log_dir_child):
            os.makedirs(self.log_dir_child)

    # Continuously draw. 19 steps each trajectory
    # Play on cpu
    def play(self, storage=None):
        self._reset_trajectory()

        while self.n < 50:
            self.a2c.set_weights(ray.get(storage.get_info.remote('a2c_weights')))

            self.time_step.observation["noise_sample"] = self.noise_sample

            # for key in self.trajectory['prev_actions']:
            #     self.trajectory['prev_actions'][key].append(self.state.prev_action[key])

            with torch.no_grad():
                # action is a dictionary
                agent_out, self.state = self.a2c.PI(self.time_step.step_type, self.time_step.observation, self.state)

            # get action from
            action = agent_out.action
            obs = self.time_step.observation["canvas"]
            if obs.shape[2] != 3:
                obs = np.repeat(obs, 3, axis=2)
            self.trajectory['images'].append(obs)  # canvas is (64, 64, 3)
            save_image(torch.tensor(obs).permute(2,0,1), self.log_dir_child+f'/{self.steps}.png', normalize=True)

            self.trajectory['values'].append(agent_out.baseline.numpy())  # baseline is tensor(value)

            self.time_step = self.env.step(action)
            self.steps += 1

            for key in self.trajectory['actions']:
                self.trajectory['actions'][key].append(action[key])
                self.trajectory['prev_actions'][key].append(action[key])
                self.trajectory['action_masks'][key].append(self.time_step.observation['action_mask'][key])

            # 19 steps check
            if self.steps >= self.config.n_paint_steps:
                final_render = self.time_step.observation["canvas"]  # final render for reward
                if final_render.shape[2] != 3:
                    final_render = np.repeat(final_render, 3, axis=2)
                self.trajectory['final_render'] = final_render
                self.trajectory['label'] = self.label.item()



                for key in self.trajectory['prev_actions']:
                    self.trajectory['prev_actions'][key] = self.trajectory['prev_actions'][key][
                                                           :-1]  # each action is (time,)

                storage.save_trajectory.remote(self.trajectory, final_render, self.label.item())
                storage.increment_n_games.remote()
                self.n += 1
                self._reset_trajectory()


cfg = SpiralConfig()
checkpoint = torch.load(cfg.checkpoint_path)
checkpoint['terminate'] = False
storage = SharedStorage.remote(checkpoint, cfg)
painters = [Demo_Painter.remote(cfg, i) for i in range(1, 6)]
[painter.play.remote(storage=storage) for painter in painters]

time.sleep(1000000)
