from __future__ import annotations

import math
import os
import shutil
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import torch

from .hooks import DEFAULT_HOOKS, Hook
from .optim_utils import build_optimizer
from .scheduler_utils import build_scheduler, describe_scheduler
from .train_utils import count_parameters, move_targets_to_device, save_json


@dataclass
class TrainerState:
    epoch: int = 0
    inner_iter: int = 0
    global_step: int = 0
    did_optimizer_step: bool = False
    best_metric: Optional[float] = None
    latest_val_metrics: Dict[str, float] = field(default_factory=dict)
    train_meter_window: int = 50
    train_history: Dict[str, Deque[float]] = field(default_factory=dict)
    last_iter_time: float = 0.0
    _run_start_time: Optional[float] = None
    _iter_start_time: Optional[float] = None

    def update_train_metric(self, name: str, value: float) -> None:
        if name not in self.train_history:
            self.train_history[name] = deque(maxlen=self.train_meter_window)
        self.train_history[name].append(float(value))

    def mean_train_metrics(self) -> Dict[str, float]:
        return {k: (sum(v) / max(1, len(v))) for k, v in self.train_history.items() if len(v) > 0}


class SegTrainer:
    """
    Hook-based trainer inspired by MMEngine/Detectron2 style training loops, but implemented with
    Hugging Face Accelerate for stable single-GPU, DDP, and DeepSpeed support.
    """

    def __init__(
        self,
        cfg,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        accelerator,
        logger,
        hooks: Optional[List[Hook]] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.accelerator = accelerator
        self.logger = logger
        self.state = TrainerState()
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / 'checkpoints'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer, self.optimizer_summary = build_optimizer(self.model, self.cfg, logger=self.logger)
        total_update_steps = self._compute_total_update_steps()
        self.scheduler = build_scheduler(self.optimizer, self.cfg, total_update_steps)
        self.logger.info(describe_scheduler(self.cfg, total_update_steps))

        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )
        self.total_update_steps = total_update_steps
        self.hooks = sorted(hooks or DEFAULT_HOOKS, key=lambda h: getattr(h, 'priority', 50))
        self.param_summary = count_parameters(self.accelerator.unwrap_model(self.model))

    def _compute_total_update_steps(self) -> int:
        if self.cfg.max_steps and self.cfg.max_steps > 0:
            return int(self.cfg.max_steps)
        steps_per_epoch = math.ceil(len(self.train_loader) / max(1, self.cfg.gradient_accumulation_steps))
        return int(max(1, steps_per_epoch * self.cfg.epochs))

    def is_main_process(self) -> bool:
        return bool(self.accelerator.is_main_process)

    def is_better_metric(self, current: float) -> bool:
        if self.state.best_metric is None:
            return True
        if str(self.cfg.monitor_mode).lower() == 'max':
            return current > self.state.best_metric
        return current < self.state.best_metric

    def _call_hooks(self, fn_name: str) -> None:
        for hook in self.hooks:
            fn = getattr(hook, fn_name, None)
            if fn is not None:
                fn(self)

    def _reduce_loss_dict(self, losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
        reduced = {}
        for k, v in losses.items():
            if not torch.is_tensor(v):
                continue
            rv = self.accelerator.reduce(v.detach(), reduction='mean')
            reduced[k] = float(rv.item())
        return reduced

    def log_train_metrics(self) -> None:
        metrics = self.state.mean_train_metrics()
        if not metrics:
            return
        lr = self.optimizer.param_groups[0]['lr']
        payload = {'train/lr': float(lr), 'meta/epoch': self.state.epoch + 1, 'meta/step': self.state.global_step}
        for k, v in metrics.items():
            payload[f'train/{k}'] = float(v)
        if self.state.last_iter_time > 0:
            payload['train/iter_time'] = float(self.state.last_iter_time)
        self.logger.info('step=%d | %s', self.state.global_step, ' | '.join(f'{k}={v:.4f}' for k, v in payload.items() if isinstance(v, float)))
        self.accelerator.log(payload, step=self.state.global_step)

    def log_validation_metrics(self, metrics: Dict[str, float]) -> None:
        payload = {'meta/step': self.state.global_step, 'meta/epoch': self.state.epoch + 1}
        payload.update(metrics)
        self.logger.info('validation @ step=%d | %s', self.state.global_step, ' | '.join(f'{k}={v:.4f}' for k, v in metrics.items()))
        self.accelerator.log(payload, step=self.state.global_step)

    def save_checkpoint(self, tag: str) -> None:
        ckpt_path = self.ckpt_dir / tag
        if self.accelerator.is_main_process and ckpt_path.exists():
            shutil.rmtree(ckpt_path)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(ckpt_path)
        if self.accelerator.is_main_process:
            trainer_state = {
                'epoch': self.state.epoch,
                'global_step': self.state.global_step,
                'best_metric': self.state.best_metric,
                'monitor': self.cfg.monitor,
                'monitor_mode': self.cfg.monitor_mode,
            }
            save_json(trainer_state, ckpt_path / 'trainer_state.json')
            self._prune_checkpoints()
            self.logger.info(f'Saved checkpoint to {ckpt_path}')
        self.accelerator.wait_for_everyone()

    def _prune_checkpoints(self) -> None:
        if self.cfg.keep_last_k <= 0:
            return
        ckpts = [p for p in self.ckpt_dir.iterdir() if p.is_dir() and p.name not in {'best'}]
        ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime, reverse=True)
        for p in ckpts[self.cfg.keep_last_k:]:
            shutil.rmtree(p, ignore_errors=True)

    def resume_if_needed(self) -> None:
        if not self.cfg.resume_from:
            return
        resume_path = Path(self.cfg.resume_from)
        self.accelerator.print(f'Resuming from {resume_path}')
        self.accelerator.load_state(resume_path)
        state_file = resume_path / 'trainer_state.json'
        if state_file.exists():
            import json
            with open(state_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.state.epoch = int(meta.get('epoch', 0))
            self.state.global_step = int(meta.get('global_step', 0))
            self.state.best_metric = meta.get('best_metric', None)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        self._call_hooks('before_val_epoch')
        totals: Dict[str, float] = defaultdict(float)
        counts = 0
        for images, targets in self.val_loader:
            images = images.to(self.accelerator.device, non_blocking=True)
            targets = move_targets_to_device(targets, self.accelerator.device)
            _, losses = self.model(images, targets)
            reduced = self._reduce_loss_dict(losses)
            total_loss = float(sum(reduced.values()))
            reduced['val/total_loss'] = total_loss
            for k, v in reduced.items():
                key = k if k.startswith('val/') else f'val/{k}'
                totals[key] += float(v)
            counts += 1
            if self.cfg.debug and counts >= 5:
                break
        metrics = {k: v / max(1, counts) for k, v in totals.items()}
        self.state.latest_val_metrics = metrics
        self._call_hooks('after_val_epoch')
        self.model.train()
        return metrics

    def train(self) -> None:
        self._call_hooks('before_run')
        if self.accelerator.is_main_process:
            self.logger.info('Parameter summary: %s', self.param_summary)
            self.logger.info('Optimizer groups: %s', self.optimizer_summary['groups'])

        self.resume_if_needed()
        for epoch in range(self.state.epoch, self.cfg.epochs):
            self.state.epoch = epoch
            self._call_hooks('before_train_epoch')
            self.model.train()

            for images, targets in self.train_loader:
                if self.cfg.max_steps > 0 and self.state.global_step >= self.cfg.max_steps:
                    break

                self._call_hooks('before_train_iter')
                self.state.did_optimizer_step = False
                images = images.to(self.accelerator.device, non_blocking=True)
                targets = move_targets_to_device(targets, self.accelerator.device)

                with self.accelerator.accumulate(self.model):
                    outputs, losses = self.model(images, targets)
                    loss = sum(losses.values())
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients and self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.accelerator.sync_gradients:
                        self.state.did_optimizer_step = True
                        self.state.global_step += 1

                reduced = self._reduce_loss_dict(losses)
                reduced['total_loss'] = float(sum(reduced.values()))
                for k, v in reduced.items():
                    self.state.update_train_metric(k, v)

                self._call_hooks('after_train_iter')

                if self.cfg.val_interval > 0 and self.state.global_step > 0 and self.state.global_step % self.cfg.val_interval == 0:
                    self.validate()

                if self.cfg.debug and self.state.global_step >= 5:
                    break

            self._call_hooks('after_train_epoch')
            if self.cfg.max_steps > 0 and self.state.global_step >= self.cfg.max_steps:
                break
            if self.cfg.debug and self.state.global_step >= 5:
                break

        self._call_hooks('after_run')
        self.accelerator.end_training()