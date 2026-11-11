
"""
Optimizer and learning rate scheduler.

Author: Mashego Gideon Mabeloane
"""

import torch


def get_optimizer(model):
    # Only pass parameters that require gradients
    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=5e-5,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=3,
        threshold=1e-3,
        min_lr=1e-7,
    )

    return optimizer, scheduler
