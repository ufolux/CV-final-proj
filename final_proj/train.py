import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import collections
from storage import SharedStorage
from painter import Painter
from learners import DiscLearner, PolicyLearner

from networks import Discriminator
from config import SpiralConfig
from policies import A2C


def train_loop():



def train():
    cfg = SpiralConfig()
    if cfg.checkpoint_path is not None:
        checkpoint = torch.load(cfg.checkpoint_path)
        checkpoint['terminate'] = False
    else:
        checkpoint = {
            "a2c_weights": None,
            "d_weights": None,
            "optimizer_state": None,
            "a2c_training_steps": 0,
            "d_training_steps": 0,
            "lr": 0,
            "num_played_games": 0,
            "terminate": False,
            "loss":None,
            "policy_loss":None,
            "value_loss":None,
            "entropy_loss":None,
            "d_loss":None,
            "d_fake":None,
            "d_real":None,
            "reward":None,
            "value":None,
            "render_sample":None,
            "real_sample":None,
            "fake_sample":None,
            "neg_log":None,
            "adv":None,
            "grad_norm":None,
            "d_std":None,
            "d_mean":None,
            "n_batches_skipped":0,
        }

    storage = SharedStorage(checkpoint, cfg)
    painter = Painter(cfg)
    discriminator = Discriminator()
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=cfg.d_lr, betas=(0.5, 0.999))
    a2c = A2C(cfg.action_spec, input_shape=cfg.input_shape, grid_shape=cfg.grid_shape, action_order="libmypaint", cuda=True)
    a2c_optimizer = torch.optim.Adam(a2c.parameters(), lr=cfg.a2c_lr)




