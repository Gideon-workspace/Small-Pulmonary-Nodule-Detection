"""
Optimizer and learning rate scheduler.

Author: Mashego Gideon Mabeloane
"""

import torch


def get_optimizer(model):
    # Only pass parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-4,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1,
    )

    return optimizer, scheduler
