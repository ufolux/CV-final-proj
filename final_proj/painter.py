

"""
class Painter:
    def __init__(self, config, seed=7):
        self.config = config
        self.env = self._init_env()
        self._action_spec = config.action_spec
        np.random.seed(seed)
        self.steps = 0

    def _init_env(self):
        env = libmypaint.LibMyPaint(**self.config.libmypaint_params)
        return env

    def _reset_trajectory(self, a2c_init_state):
        self.time_step = self.env.reset()
        self.noise_sample = torch.randn(1, 10)
        self.state = a2c_init_state
        self.steps = 0
        self.trajectory = {
            'images': [],
            'action_masks': collections.OrderedDict([(spec, [self.time_step.observation['action_mask'][spec]]) for spec in self._action_spec]),
            'prev_actions': collections.OrderedDict([(spec, [self.state.prev_action[spec]]) for spec in self._action_spec]),
            'actions': collections.OrderedDict([(spec, []) for spec in self._action_spec]),
            'values': [],
            'noise_sample': self.noise_sample.squeeze().numpy(),
        }

    # Continuously draw. 19 steps each trajectory
    # Play on cpu
    def play(self, a2c, storage=None):
        a2c.eval()
        self._reset_trajectory(a2c.initial_state())

        while self.steps <= self.config.n_paint_steps:
            self.time_step.observation["noise_sample"] = self.noise_sample
            with torch.no_grad():
                # action is a dictionary
                agent_out, self.state = a2c.PI(self.time_step.step_type, self.time_step.observation, self.state)

            # get action from
            action = agent_out.action
            obs = self.time_step.observation["canvas"]
            if obs.shape[2] != 3:
                obs = np.repeat(obs, 3, axis=2)
            self.trajectory['images'].append(obs) # canvas is (64, 64, 3)
            self.trajectory['values'].append(agent_out.baseline.numpy()) # baseline is tensor(value)

            self.time_step = self.env.step(action)
            self.steps += 1

            for key in self.trajectory['actions']:
                self.trajectory['actions'][key].append(action[key])
                self.trajectory['prev_actions'][key].append(action[key])
                self.trajectory['action_masks'][key].append(self.time_step.observation['action_mask'][key])

        final_render = self.time_step.observation["canvas"] # final render for reward
        if final_render.shape[2] != 3:
            final_render = np.repeat(final_render, 3, axis=2)
        self.trajectory['final_render'] = final_render

        for key in self.trajectory['prev_actions']:
            self.trajectory['prev_actions'][key] = self.trajectory['prev_actions'][key][:-1] # each action is (time,)

        return self.trajectory


if __name__ == '__main__':
    cfg = SpiralConfig()
    painter = Painter(cfg)
    a2c = A2C(cfg.action_spec, input_shape=cfg.input_shape, grid_shape=cfg.grid_shape, action_order="libmypaint")
    weight = copy.deepcopy(a2c.get_weights())
    a2c = a2c.to('cpu')
    trajs = painter.play(a2c)
    for i, img in enumerate(trajs['images']):
        plt.imshow(img)
        plt.savefig(f'{i}.png')

    trajs = painter.play(a2c)
    for i, img in enumerate(trajs['images']):
        plt.imshow(img)
        plt.savefig(f'{i}.png')

    trajs = painter.play(a2c)
    for i, img in enumerate(trajs['images']):
        plt.imshow(img)
        plt.savefig(f'{i}.png')

    trajs = painter.play(a2c)
    for i, img in enumerate(trajs['images']):
        plt.imshow(img)
        plt.savefig(f'{i}.png')
    print()
"""

import ray
import collections
import numpy as np
import torch
from policies import A2C
import spiral.environments.libmypaint as libmypaint
from config import SpiralConfig


class Painter():
    def __init__(self, config, seed=7):
        self.config = config
        self.env = self._init_env()
        self._action_spec = config.action_spec
        np.random.seed(seed)

        # init policy on cpu
        self.a2c = A2C(self.env.action_spec(), input_shape=config.input_shape, grid_shape=config.grid_shape,
                       action_order="libmypaint")
        # self.a2c.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.a2c.to('cpu')
        self.a2c.eval()

        self.steps = 0

        self._reset_trajectory()

    def _init_env(self):
        env = libmypaint.LibMyPaint(**self.config.libmypaint_params)
        return env

    def _reset_trajectory(self):
        self.time_step = self.env.reset()
        self.noise_sample = torch.randn(1, 10)
        self.state = self.a2c.initial_state()
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

    # Continuously draw. 19 steps each trajectory
    # Play on cpu
    def play(self, weight, storage=None):
        self._reset_trajectory()

        while True:
            self.a2c.set_weights(weight)

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

            self.trajectory['values'].append(agent_out.baseline.numpy())  # baseline is tensor(value)

            self.time_step = self.env.step(action)
            self.steps += 1

            for key in self.trajectory['actions']:
                self.trajectory['actions'][key].append(action[key])
                self.trajectory['prev_actions'][key].append(action[key])
                self.trajectory['action_masks'][key].append(self.time_step.observation['action_mask'][key])


            """
            # 19 steps check
            if self.steps >= self.config.n_paint_steps:
                final_render = self.time_step.observation["canvas"]  # final render for reward
                if final_render.shape[2] != 3:
                    final_render = np.repeat(final_render, 3, axis=2)
                self.trajectory['final_render'] = final_render

                # for key in self.trajectory['actions']:
                #     self.trajectory['actions'][key] = np.stack(self.trajectory['actions'][key]) # each action is (time,)
                #     self.trajectory['prev_actions'][key] = np.stack(self.trajectory['prev_actions'][key][:-1]) # each action is (time,)

                for key in self.trajectory['prev_actions']:
                    self.trajectory['prev_actions'][key] = self.trajectory['prev_actions'][key][
                                                           :-1]  # each action is (time,)

                storage.save_trajectory.remote(self.trajectory, final_render)
                storage.increment_n_games.remote()
                self._reset_trajectory()
            """

if __name__ == '__main__':
    cfg = SpiralConfig()
    a2c = A2C(cfg.action_spec, input_shape=cfg.input_shape, grid_shape=cfg.grid_shape, action_order="libmypaint")
    painter = Painter(cfg)
    painter.play()