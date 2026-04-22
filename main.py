from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Dict, Optional

import yaml

from train.config import FullConfig, dump_config, load_config
from train.utils import build_datasets_and_loaders, seed_everything, setup_logging
from train.trainer import SegTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train VAW open-vocabulary segmentation baseline with Accelerate.')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config.')
    parser.add_argument('--train_annotation_json', type=str, default=None)
    parser.add_argument('--val_annotation_json', type=str, default=None)
    parser.add_argument('--image_root', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--val_batch_size', type=int, default=None)
    parser.add_argument('--input_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)
    parser.add_argument('--mixed_precision', type=str, default=None)
    parser.add_argument('--dataset_pipeline', type=str, default=None)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--use_wandb', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def build_overrides(args) -> Dict:
    train = {}
    for key in [
        'train_annotation_json','val_annotation_json','image_root','output_dir','exp_name',
        'train_batch_size','val_batch_size','input_size','epochs','max_steps','num_workers',
        'lr','weight_decay','gradient_accumulation_steps','mixed_precision','dataset_pipeline',
        'resume_from','wandb_project','wandb_run_name'
    ]:
        value = getattr(args, key)
        if value is not None:
            train[key] = value
    if args.use_wandb is not None:
        train['use_wandb'] = bool(args.use_wandb)
    if args.debug:
        train['debug'] = True
    return {'train': train}


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=build_overrides(args))

    try:
        from accelerate import Accelerator, DistributedDataParallelKwargs
        from accelerate.utils import ProjectConfiguration, set_seed
    except Exception as e:
        raise ImportError(
            'This trainer requires `accelerate`. Please install `accelerate`, `wandb`, and optionally `deepspeed`.') from e

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.train.find_unused_parameters)
    project_config = ProjectConfiguration(project_dir=cfg.train.output_dir, logging_dir=str(Path(cfg.train.output_dir) / 'logs'))

    log_with = 'wandb' if cfg.train.use_wandb else None
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        mixed_precision=cfg.train.mixed_precision,
        log_with=log_with,
        project_config=project_config,
        kwargs_handlers=[kwargs],
    )

    logger = setup_logging(cfg.train.output_dir, rank=accelerator.process_index)
    seed_everything(cfg.train.seed + accelerator.process_index)
    try:
        set_seed(cfg.train.seed, device_specific=True)
    except Exception:
        pass

    if accelerator.is_main_process:
        dump_config(cfg, Path(cfg.train.output_dir) / 'resolved_config.yaml')

    train_dataset, val_dataset, train_loader, val_loader = build_datasets_and_loaders(cfg.train, accelerator=accelerator)

    from models.model import build_model_from_dataset
    model = build_model_from_dataset(train_dataset, cfg=cfg.model)

    tracker_config = yaml.safe_load(yaml.safe_dump({'train': vars(cfg.train)}))
    if cfg.train.use_wandb:
        accelerator.init_trackers(
            project_name=cfg.train.wandb_project,
            config=tracker_config,
            init_kwargs={'wandb': {
                'name': cfg.train.wandb_run_name or cfg.train.exp_name,
                'entity': cfg.train.wandb_entity,
                'tags': cfg.train.wandb_tags,
            }},
        )
        if cfg.train.wandb_watch and accelerator.is_main_process:
            try:
                import wandb
                wandb.watch(model, log='gradients', log_freq=max(100, cfg.train.log_interval))
            except Exception as e:
                logger.warning(f'wandb.watch failed: {e}')

    trainer = SegTrainer(
        cfg=cfg.train,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        accelerator=accelerator,
        logger=logger,
    )
    trainer.train()


if __name__ == '__main__':
    main()