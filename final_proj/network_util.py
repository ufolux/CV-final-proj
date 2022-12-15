import time
import torch.autograd as autograd
from torch.autograd import Variable
from networks import Discriminator
from image_loader import get_image_loader_dict
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import copy
from random import sample
import ray
import collections
import numpy as np
import torch
from policies import A2C
import spiral.environments.libmypaint as libmypaint



@ray.remote(num_gpus=0.5)
class DiscLearner:
    def __init__(self, config, checkpoint=None):
        print('discriminator using gpu: ', ray.get_gpu_ids())
        self.config = config
        self.batch_size = config.batch_size
        self.discriminator = Discriminator()
        if checkpoint is not None:
            self.discriminator.set_weights(checkpoint['d_weights'])
        self.discriminator.cuda()

        self.optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=config.d_lr)
        if checkpoint is not None:
            self.discriminator_step = checkpoint['d_training_steps']
        else:
            self.discriminator_step = 0

        self.training_steps_ratio = config.training_steps_ratio

        self.real_image_loader = get_image_loader_dict(dataset=config.dataset, batch_size=config.batch_size)

    def _compute_gradient_penalty(self, D, real_samples, fake_samples, labels):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        labels = labels.long()
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates, labels).squeeze().unsqueeze(1)
        fake = Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _wgan_update(self, fake_images, fake_labels):
        real_images = torch.zeros(fake_images.size(0), 64, 64, 3).float()
        count = 0

        for i in range(1, 6):
            fake_label_i = torch.where(fake_labels == i, 1, 0).cpu()
            fake_label_i_index = torch.nonzero(fake_label_i).squeeze()

            real_images_i, label = next(self.real_image_loader[f'{i}'])

            real_images[fake_label_i_index] = real_images_i[:torch.sum(fake_label_i)].float()
            count += torch.sum(fake_label_i).item()

        if count != fake_images.size(0):
            print("Real Images didn't load correctly")

        real_sample = real_images[0].numpy()
        real_images = real_images.cuda().float()
        real_labels = fake_labels
        real_images = real_images.permute(0, 3, 1, 2)

        self.optimizer.zero_grad()
        gradient_penalty = self._compute_gradient_penalty(self.discriminator, real_images.data, fake_images.data,
                                                          fake_labels.data)
        fake_scores = self.discriminator(fake_images, fake_labels)
        real_scores = self.discriminator(real_images, real_labels)
        d_fake = fake_scores.mean()
        d_real = real_scores.mean()
        loss = d_fake - d_real + 10 * gradient_penalty

        loss.backward()
        # nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.1)
        self.optimizer.step()

        scores = torch.cat([fake_scores, real_scores], dim=0).detach()
        d_std = scores.std()
        d_mean = scores.mean()
        return loss, d_fake, d_real, d_std, d_mean, real_sample

    def learn(self, storage):
        while not ray.get(storage.is_buffer_ready.remote()):
            time.sleep(0.1)

        batch_future = storage.get_final_render_batch.remote()

        while not ray.get(storage.get_info.remote("terminate")):
            # image_batch: (batch, 64, 64, 3)
            image_batch, label_batch = ray.get(batch_future)
            batch_future = storage.get_final_render_batch.remote()

            image_batch = np.array(image_batch)
            label_batch = np.array(label_batch)
            fake_sample = image_batch[0]
            fake_label = label_batch[0]
            image_batch = torch.cuda.FloatTensor(image_batch)
            label_batch = torch.cuda.LongTensor(label_batch)
            image_batch = image_batch.permute(0, 3, 1, 2)

            loss, d_fake, d_real, d_std, d_mean, real_sample = self._wgan_update(image_batch, label_batch)

            self.discriminator_step += 1

            if self.discriminator_step % self.config.weight_copy_interval == 0:
                info = {
                    "d_weights": copy.deepcopy(self.discriminator.get_weights()),
                    "d_training_steps": self.discriminator_step,
                }

                if self.discriminator_step < 15000:  # freeze std and mean when stable
                    info["d_std"] = d_std.cpu().numpy()
                    info["d_mean"] = d_mean.cpu().numpy()
                storage.set_info.remote(info)
            if self.discriminator_step % self.config.log_interval == 0:
                storage.set_info.remote(
                    {
                        "d_loss": loss.detach().cpu().numpy(),
                        "d_fake": d_fake.detach().cpu().numpy(),
                        "d_real": d_real.detach().cpu().numpy(),
                    }
                )
            if self.discriminator_step % self.config.log_draw_interval == 0:
                storage.set_info.remote(
                    {
                        "real_sample": real_sample,
                        "fake_sample": fake_sample,
                    }
                )

            if self.training_steps_ratio is not None:
                a2c_training_steps = ray.get(storage.get_info.remote("a2c_training_steps"))
                while a2c_training_steps > 0 and self.discriminator_step // a2c_training_steps > self.training_steps_ratio:
                    a2c_training_steps = ray.get(storage.get_info.remote("a2c_training_steps"))
                    time.sleep(0.2)


@ray.remote(num_gpus=0.5)
class PolicyLearner:
    def __init__(self, config, checkpoint=None):
        print('policy using gpu: ', ray.get_gpu_ids())
        self.batch_size = config.batch_size
        self.config = config

        if checkpoint is not None:
            self.training_step = checkpoint['a2c_training_steps']
        else:
            self.training_step = 0

        self.a2c = A2C(config.action_spec, input_shape=config.input_shape, grid_shape=config.grid_shape,
                       action_order="libmypaint", cuda=True)
        if checkpoint is not None:
            self.a2c.set_weights(checkpoint['a2c_weights'])
        self.a2c.cuda()

        self.optimizer = torch.optim.RMSprop(
            self.a2c.parameters(),
            lr=self.config.a2c_lr,
        )

        self.discriminator = Discriminator()
        self.discriminator.eval()

        self.reward_mode = config.reward_mode

        self.n_batches_skipped = 0

    def _get_rewards(self, final_renders, label_batch, storage, gamma=0.99):
        # final_renders: (batch, 64, 64, 3)
        if self.reward_mode == 'l2':
            real_images = next(self.real_image_loader).numpy()
            reward = -np.sqrt(np.sum(np.square(final_renders - real_images), axis=(1, 2, 3)))
            reward = reward[:, np.newaxis]  # (batch, 1)
            reward = (reward + 50) / 50
        else:

            info = ray.get(storage.get_info.remote(['d_std', 'd_mean', 'd_weights']))
            self.discriminator.set_weights(info['d_weights'])

            with torch.no_grad():
                final_renders_tensor = torch.FloatTensor(final_renders)
                label_tensor = torch.tensor(label_batch)
                final_renders_tensor = final_renders_tensor.permute(0, 3, 1, 2)
                reward = self.discriminator(final_renders_tensor, label_tensor).squeeze()  # (batch,)
                reward = reward.numpy()[:, np.newaxis]  # (batch, 1)
                reward = (reward - info['d_mean']) / info['d_std']
                # reward = reward / 150

        return reward

    def _get_adv(self, R, V):
        return R - V

    def learn(self, storage):
        while ray.get(storage.get_info.remote("d_training_steps")) < 1:
            time.sleep(0.1)

        while not ray.get(storage.is_queue_ready.remote()):
            time.sleep(0.1)
        batch_future = storage.get_trajectory_batch.remote()
        while self.training_step < self.config.training_steps and not ray.get(storage.get_info.remote("terminate")):
            # action_batch should be a dict, each value is (batch, time, size[i]), where i comes from LOCATION_KEYS
            image_batch, action_batch, prev_action_batch, action_masks, value_batch, noise_samples, final_renders, label_batch = ray.get(
                batch_future)

            image_batch = np.array(image_batch)
            label_batch = np.array(label_batch)
            # final_renders = image_batch[:, -1, :, :, :] # (batch, 64, 64, 3)
            final_renders = np.array(final_renders)
            reward_batch = self._get_rewards(final_renders, label_batch, storage)  # (batch,)
            reward_mean = np.mean(reward_batch)

            reward_batch = np.repeat(reward_batch, image_batch.shape[1], axis=1)  # (batch, time)

            value_mean = np.mean(value_batch)
            adv_batch = self._get_adv(reward_batch, value_batch)  # (batch, time)

            noise_samples = noise_samples[:, np.newaxis, :]  # (batch, 1, 10)
            noise_samples = np.repeat(noise_samples, image_batch.shape[1], axis=1)  # (batch, time, 10)

            image_batch = torch.cuda.FloatTensor(image_batch)
            label_batch = torch.cuda.LongTensor(label_batch)
            reward_batch = torch.cuda.FloatTensor(reward_batch)
            adv_batch = torch.cuda.FloatTensor(adv_batch)
            noise_samples = torch.cuda.FloatTensor(noise_samples)

            results = self.a2c.optimize(
                image_batch,
                prev_action_batch,
                action_batch,
                reward_batch,
                adv_batch,
                action_masks,
                noise_samples,
                F.one_hot(label_batch.squeeze() % 5, num_classes=5).float(),
                entropy_weight=self.config.entropy_weight,
                value_weight=self.config.value_weight,
            )

            if results is None:  # numerical issue
                self.n_batches_skipped += 1
            else:
                total_loss, policy_loss, value_loss, entropy_loss, neg_log = results

                self.optimizer.zero_grad()
                total_loss.backward()
                # nn.utils.clip_grad_norm_(self.a2c.parameters(), 0.5)
                self.optimizer.step()

                self.training_step += 1

                if self.training_step % self.config.weight_copy_interval == 0:
                    storage.set_info.remote(
                        {
                            "a2c_weights": copy.deepcopy(self.a2c.get_weights()),
                            "a2c_training_steps": self.training_step,
                        }
                    )

                if self.training_step % self.config.log_interval == 0:
                    entropy_sample = entropy_loss.detach().cpu().numpy()
                    if entropy_sample < 3.0:  # stop training if entropy collapsed
                        storage.set_info.remote("terminate", True)
                    else:
                        grad_norm = 0
                        for p in self.a2c.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.detach().data.norm(2)
                                grad_norm += param_norm.item() ** 2
                        grad_norm = grad_norm ** 0.5

                        storage.set_info.remote(
                            {
                                "loss": total_loss.detach().cpu().numpy(),
                                "policy_loss": policy_loss.detach().cpu().numpy(),
                                "value_loss": value_loss.detach().cpu().numpy(),
                                "entropy_loss": entropy_sample,
                                "reward": reward_mean,
                                "value": value_mean,
                                "neg_log": neg_log.detach().cpu().numpy(),
                                "adv": adv_batch.mean().cpu().numpy(),
                                "grad_norm": grad_norm,
                                "n_batches_skipped": self.n_batches_skipped,
                            }
                        )

                if self.training_step % self.config.log_draw_interval == 0:
                    storage.set_info.remote(
                        {
                            "render_sample": final_renders[0],
                        }
                    )
                    save_image(torch.tensor(final_renders[:25]).permute(0, 3, 1, 2),
                               (self.config.results_path + "%d.png") % self.training_step, nrow=5, normalize=True)

                if self.training_step % self.config.checkpoint_interval == 0:
                    storage.save_checkpoint.remote(savename=str(self.training_step))

                # if self.training_step < 2000:
                #     time.sleep(1)
                # if self.training_step < 5000:
                #     time.sleep(0.5)

            while not ray.get(storage.is_queue_ready.remote()):
                time.sleep(0.1)
            batch_future = storage.get_trajectory_batch.remote()


@ray.remote
class SharedStorage:
    def __init__(self, checkpoint, config):
        self.config = config
        self._action_spec = config.action_spec
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        self.total_size = self.buffer_size * self.batch_size
        self.trajectories = [] # queue for policy learner
        self.images_rb = [None for _ in range(self.total_size)]
        self.image_label = [None for _ in range(self.total_size)]
        self.pointer = 0
        self.image_pointer = 0
        self.images_rb_len = 0

        self.current_checkpoint = copy.deepcopy(checkpoint)

    def get_tragectories(self):
        return self.trajectories

    def get_images(self):
        return np.array(self.images_rb)

    def save_trajectory(self, trajectory, final_render, label):
        self.trajectories.append(trajectory)
        # self.noise_samples[self.pointer] = noise_sample

        self.images_rb[self.pointer] = final_render
        self.image_label[self.pointer] = label
        # if not self.buffer_ready and self.pointer == self.batch_size-1:
        #     self.buffer_ready = True
        self.pointer = (self.pointer + 1) % self.total_size

    def is_buffer_ready(self):
        return self.pointer >= self.batch_size or self.images_rb[-1] is not None

    def is_queue_ready(self):
        return len(self.trajectories) >= self.batch_size

    def increment_n_games(self):
        self.set_info('num_played_games', self.current_checkpoint['num_played_games']+1)

    def get_trajectory_batch(self):
        image_batch = []
        value_batch = []
        action_batch = collections.OrderedDict([(spec, []) for spec in self._action_spec])
        prev_action_batch = collections.OrderedDict([(spec, []) for spec in self._action_spec])
        action_mask_batch = collections.OrderedDict([(spec, []) for spec in self._action_spec])

        traj_batch = self.trajectories[:self.batch_size]
        self.trajectories = self.trajectories[self.batch_size:]

        noise_batch = [t['noise_sample'] for t in traj_batch]
        render_batch = [t['final_render'] for t in traj_batch]
        label_batch = [t['label'] for t in traj_batch]

        for t in traj_batch:
            image_batch.append(np.array(t['images']))
            for key in self._action_spec:
                action_batch[key].append(t['actions'][key])
                prev_action_batch[key].append(t['prev_actions'][key])
                action_mask_batch[key].append(t['action_masks'][key])
            # action_batch.append(t['actions'])
            value_batch.append(t['values'])

        # lastly stack actions
        for key in self._action_spec:
            action_batch[key] = np.stack(action_batch[key])
            prev_action_batch[key] = np.stack(prev_action_batch[key])
            action_mask_batch[key] = np.stack(action_mask_batch[key])

        # image_batch: (batch, time, 64, 64, 3)
        # each action should be (batch, time)
        # value: (batch, time)
        return np.array(image_batch), action_batch,\
            prev_action_batch, action_mask_batch,\
            np.array(value_batch), np.array(noise_batch), np.array(render_batch), np.array(label_batch)

    def get_final_render_batch(self):
        if self.images_rb[-1] is None: # buffer not full
            sampled = range(self.pointer)
        else:
            sampled = range(self.total_size)
        sampled = sample(sampled, self.batch_size)
        sampled_images = [self.images_rb[i] for i in sampled]
        sampled_labels = [self.image_label[i] for i in sampled]
        return np.array(sampled_images), np.array(sampled_labels)


    def save_checkpoint(self, savename=None):
        if not savename:
            savename = "model"
        savename = os.path.join(self.config.results_path, savename+'.checkpoint')

        torch.save(self.current_checkpoint, savename)

    def get_checkpoint(self):
        return copy.deepcopy(self.curren_ctheckpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint.get(keys, 0)
        elif isinstance(keys, list):
            return {key: self.current_checkpoint.get(key, 0) for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        # if keys in ('d_std', 'd_mean') and self.current_checkpoint.get(keys, None) is not None:
        #     self.current_checkpoint[keys] = self.current_checkpoint[keys] * 0.999 + self.current_checkpoint[keys] * 0.001
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError


@ray.remote
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
        self.noise_sample_no_label = torch.randn(1, 10)
        self.label = torch.randint(1, 6, (1,))
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

    # Continuously draw. 19 steps each trajectory
    # Play on cpu
    def play(self, storage=None):
        self._reset_trajectory()

        while ray.get(storage.get_info.remote('a2c_training_steps')) < self.config.training_steps and not ray.get(
                storage.get_info.remote('terminate')):
            self.a2c.set_weights(ray.get(storage.get_info.remote('a2c_weights')))

            self.time_step.observation["noise_sample"] = self.noise_sample

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
                self._reset_trajectory()

