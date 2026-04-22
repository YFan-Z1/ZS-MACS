from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


class Hook:
    priority = 50

    def before_run(self, trainer):
        pass

    def after_run(self, trainer):
        pass

    def before_train_epoch(self, trainer):
        pass

    def after_train_epoch(self, trainer):
        pass

    def before_train_iter(self, trainer):
        pass

    def after_train_iter(self, trainer):
        pass

    def before_val_epoch(self, trainer):
        pass

    def after_val_epoch(self, trainer):
        pass


class IterTimerHook(Hook):
    priority = 10

    def before_run(self, trainer):
        trainer.state._run_start_time = time.time()
        trainer.state._iter_start_time = None

    def before_train_iter(self, trainer):
        trainer.state._iter_start_time = time.time()

    def after_train_iter(self, trainer):
        if trainer.state._iter_start_time is not None:
            trainer.state.last_iter_time = time.time() - trainer.state._iter_start_time


class LoggerHook(Hook):
    priority = 60

    def after_train_iter(self, trainer):
        if trainer.state.global_step == 0:
            return
        if trainer.state.global_step % trainer.cfg.log_interval != 0:
            return
        trainer.log_train_metrics()

    def after_val_epoch(self, trainer):
        if trainer.state.latest_val_metrics:
            trainer.log_validation_metrics(trainer.state.latest_val_metrics)


class ParamSchedulerHook(Hook):
    priority = 40

    def after_train_iter(self, trainer):
        if trainer.scheduler is not None and trainer.state.did_optimizer_step:
            trainer.scheduler.step()


class CheckpointHook(Hook):
    priority = 70

    def after_train_iter(self, trainer):
        if trainer.cfg.checkpoint_interval > 0 and trainer.state.global_step > 0:
            if trainer.state.global_step % trainer.cfg.checkpoint_interval == 0:
                trainer.save_checkpoint(tag=f"step_{trainer.state.global_step:07d}")

    def after_train_epoch(self, trainer):
        if trainer.cfg.save_every_epoch:
            trainer.save_checkpoint(tag=f"epoch_{trainer.state.epoch + 1:03d}")

        if trainer.cfg.save_best and trainer.state.latest_val_metrics:
            monitor_key = trainer.cfg.monitor
            if monitor_key in trainer.state.latest_val_metrics:
                current = float(trainer.state.latest_val_metrics[monitor_key])
                if trainer.is_better_metric(current):
                    trainer.state.best_metric = current
                    trainer.save_checkpoint(tag='best')


DEFAULT_HOOKS = [
    IterTimerHook(),
    ParamSchedulerHook(),
    LoggerHook(),
    CheckpointHook(),
]