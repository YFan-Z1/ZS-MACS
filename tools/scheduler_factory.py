import math
from typing import Optional

import torch


def _resolve_total_warmup_steps(config, total_update_steps: int) -> int:
    warmup_steps = int(getattr(config, 'warmup_steps', 0) or 0)
    if warmup_steps <= 0:
        warmup_ratio = float(getattr(config, 'warmup_ratio', 0.0) or 0.0)
        warmup_steps = int(round(total_update_steps * warmup_ratio)) if warmup_ratio > 0 else 0
    return max(0, min(warmup_steps, max(0, total_update_steps - 1)))


def _linear_warmup_cosine_lambda(
    current_step: int,
    total_update_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if total_update_steps <= 0:
        return 1.0
    if warmup_steps > 0 and current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _linear_warmup_linear_decay_lambda(
    current_step: int,
    total_update_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if total_update_steps <= 0:
        return 1.0
    if warmup_steps > 0 and current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)


def _linear_warmup_constant_lambda(current_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    if current_step < warmup_steps:
        return float(current_step + 1) / float(max(1, warmup_steps))
    return 1.0


def build_scheduler(optimizer: torch.optim.Optimizer, config, total_update_steps: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Build an update-based LR scheduler.
    Supported names:
      - step
      - multistep
      - cosine
      - cosine_restart
      - linear_warmup_cosine
      - linear_warmup_linear
      - linear_warmup_constant
      - onecycle
      - none
    """
    scheduler_name = str(getattr(config, 'lr_scheduler', 'step')).lower()
    warmup_steps = _resolve_total_warmup_steps(config, total_update_steps)
    min_lr_ratio = float(getattr(config, 'min_lr_ratio', 0.0) or 0.0)

    if scheduler_name in {'none', 'constant', 'off'}:
        return None

    if scheduler_name == 'step':
        step_size = int(getattr(config, 'step_size', max(1, total_update_steps // 3)))
        gamma = float(getattr(config, 'gamma', 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, step_size), gamma=gamma)

    if scheduler_name == 'multistep':
        milestones = getattr(config, 'milestones', None)
        if not milestones:
            milestones = [int(total_update_steps * 0.6), int(total_update_steps * 0.85)]
        milestones = [max(1, int(m)) for m in milestones]
        gamma = float(getattr(config, 'gamma', 0.5))
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    if scheduler_name == 'cosine':
        eta_min_ratio = float(getattr(config, 'eta_min_ratio', min_lr_ratio) or min_lr_ratio)
        base_lr = optimizer.param_groups[0]['lr']
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_update_steps),
            eta_min=base_lr * eta_min_ratio,
        )

    if scheduler_name in {'cosine_restart', 'cosine_warm_restarts'}:
        t0 = int(getattr(config, 'cosine_t0', max(1, total_update_steps // 3)))
        tmult = int(getattr(config, 'cosine_t_mult', 2))
        eta_min_ratio = float(getattr(config, 'eta_min_ratio', min_lr_ratio) or min_lr_ratio)
        base_lr = optimizer.param_groups[0]['lr']
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, t0),
            T_mult=max(1, tmult),
            eta_min=base_lr * eta_min_ratio,
        )

    if scheduler_name == 'linear_warmup_cosine':
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _linear_warmup_cosine_lambda(step, total_update_steps, warmup_steps, min_lr_ratio),
        )

    if scheduler_name == 'linear_warmup_linear':
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _linear_warmup_linear_decay_lambda(step, total_update_steps, warmup_steps, min_lr_ratio),
        )

    if scheduler_name == 'linear_warmup_constant':
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _linear_warmup_constant_lambda(step, warmup_steps),
        )

    if scheduler_name == 'onecycle':
        max_lr = float(getattr(config, 'onecycle_max_lr', optimizer.param_groups[0]['lr']))
        pct_start = float(getattr(config, 'onecycle_pct_start', 0.1))
        div_factor = float(getattr(config, 'onecycle_div_factor', 25.0))
        final_div_factor = float(getattr(config, 'onecycle_final_div_factor', 1e4))
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=max(1, total_update_steps),
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

    raise ValueError(f'Unsupported lr_scheduler: {scheduler_name}')


def describe_scheduler(config, total_update_steps: int) -> str:
    scheduler_name = str(getattr(config, 'lr_scheduler', 'step')).lower()
    warmup_steps = _resolve_total_warmup_steps(config, total_update_steps)
    if scheduler_name in {'none', 'constant', 'off'}:
        return 'LR scheduler disabled.'
    if scheduler_name == 'step':
        return (
            f"LR scheduler: StepLR(step_size={int(getattr(config, 'step_size', max(1, total_update_steps // 3)))}, "
            f"gamma={float(getattr(config, 'gamma', 0.5))}) [update-based]."
        )
    if scheduler_name == 'multistep':
        milestones = getattr(config, 'milestones', None)
        if not milestones:
            milestones = [int(total_update_steps * 0.6), int(total_update_steps * 0.85)]
        return f"LR scheduler: MultiStepLR(milestones={milestones}, gamma={float(getattr(config, 'gamma', 0.5))}) [update-based]."
    if scheduler_name == 'cosine':
        return f"LR scheduler: CosineAnnealingLR(T_max={max(1, total_update_steps)}, eta_min_ratio={float(getattr(config, 'eta_min_ratio', getattr(config, 'min_lr_ratio', 0.0)) or 0.0):.4f}) [update-based]."
    if scheduler_name in {'cosine_restart', 'cosine_warm_restarts'}:
        return f"LR scheduler: CosineAnnealingWarmRestarts(T_0={int(getattr(config, 'cosine_t0', max(1, total_update_steps // 3)))}, T_mult={int(getattr(config, 'cosine_t_mult', 2))}) [update-based]."
    if scheduler_name == 'linear_warmup_cosine':
        return f"LR scheduler: LinearWarmupCosine(total_updates={total_update_steps}, warmup_updates={warmup_steps}, min_lr_ratio={float(getattr(config, 'min_lr_ratio', 0.0) or 0.0):.4f})."
    if scheduler_name == 'linear_warmup_linear':
        return f"LR scheduler: LinearWarmupLinearDecay(total_updates={total_update_steps}, warmup_updates={warmup_steps}, min_lr_ratio={float(getattr(config, 'min_lr_ratio', 0.0) or 0.0):.4f})."
    if scheduler_name == 'linear_warmup_constant':
        return f"LR scheduler: LinearWarmupConstant(warmup_updates={warmup_steps})."
    if scheduler_name == 'onecycle':
        return f"LR scheduler: OneCycleLR(total_updates={total_update_steps}, max_lr={float(getattr(config, 'onecycle_max_lr', 0.0) or 0.0):.6g})."
    return f"LR scheduler: {scheduler_name}."