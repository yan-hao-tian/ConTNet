import torch
from timm.scheduler.cosine_lr import CosineLRScheduler

def build_lr_scheduler(epoch, warmup_epoch, optimizer, n_iter_per_epoch):
    num_steps = int(epoch * n_iter_per_epoch)
    warmup_steps = int(warmup_epoch * n_iter_per_epoch)
  
    scheduler = CosineLRScheduler(
                    optimizer,
                    t_initial=num_steps,
                    t_mul=1.,
                    lr_min=0,
                    warmup_lr_init=0,
                    warmup_t=warmup_steps,
                    cycle_limit=1,
                    t_in_epochs=False,
                )

    return scheduler


