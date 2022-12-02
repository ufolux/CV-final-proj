import os
import ray
import collections
import copy
from random import sample
import numpy as np
import torch

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
