from __future__ import annotations

import ast
import json
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_NO_DECAY_KEYWORDS = [
    'bias',
    'bn',
    'ln',
    'norm',
    'layernorm',
    'layer_norm',
    'batchnorm',
    'batch_norm',
]


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _parse_maybe_structured(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass
        try:
            return ast.literal_eval(text)
        except Exception:
            return value
    return value


def _normalize_keywords(value: Any, field_name: str = 'keywords') -> List[str]:
    value = _parse_maybe_structured(value)
    if value is None:
        return []
    if isinstance(value, dict):
        value = value.get(field_name, [])
    return [str(v) for v in _to_list(value) if str(v) != '']


def _match_keywords(name: str, keywords: Sequence[str], match_all: bool = False) -> bool:
    if not keywords:
        return False
    if match_all:
        return all(keyword in name for keyword in keywords)
    return any(keyword in name for keyword in keywords)


def _contains_no_decay_keyword(name: str, keywords: Sequence[str]) -> bool:
    lname = name.lower()
    return any(keyword.lower() in lname for keyword in keywords)


def _parse_param_group_specs(config: Any) -> List[Dict[str, Any]]:
    raw = _get(config, 'optimizer_param_groups', None)
    if raw is None:
        raw = _get(config, 'param_group_specs', None)
    raw = _parse_maybe_structured(raw)
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise TypeError('optimizer_param_groups / param_group_specs must be a list of dicts.')

    specs: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f'Param-group spec at index {idx} must be a dict, got {type(item)}.')
        spec = dict(item)
        spec.setdefault('name', f'group_{idx}')
        spec['keywords'] = _normalize_keywords(spec.get('keywords', spec.get('include_keywords', [])))
        spec['exclude_keywords'] = _normalize_keywords(spec.get('exclude_keywords', []))
        spec['match_all'] = bool(spec.get('match_all', False))
        spec['trainable'] = bool(spec.get('trainable', True))
        specs.append(spec)
    return specs


def set_trainable_by_keywords(model: torch.nn.Module, config: Any, logger: Optional[Any] = None) -> Dict[str, List[str]]:
    reset_trainable = bool(_get(config, 'reset_trainable', False))
    use_group_keywords_as_trainable = bool(_get(config, 'use_group_keywords_as_trainable', True))

    trainable_keywords = _normalize_keywords(_get(config, 'trainable_keywords', None))
    frozen_keywords = _normalize_keywords(_get(config, 'frozen_keywords', _get(config, 'freeze_keywords', None)))
    group_specs = _parse_param_group_specs(config)

    if reset_trainable:
        for _, param in model.named_parameters():
            param.requires_grad = False

    changed_trainable: List[str] = []
    changed_frozen: List[str] = []

    if trainable_keywords:
        for name, param in model.named_parameters():
            if _match_keywords(name, trainable_keywords):
                if not param.requires_grad:
                    changed_trainable.append(name)
                param.requires_grad = True

    if group_specs and use_group_keywords_as_trainable:
        for spec in group_specs:
            if not spec.get('trainable', True):
                continue
            keywords = spec.get('keywords', [])
            exclude_keywords = spec.get('exclude_keywords', [])
            match_all = bool(spec.get('match_all', False))
            for name, param in model.named_parameters():
                if _match_keywords(name, keywords, match_all=match_all) and not _match_keywords(name, exclude_keywords):
                    if not param.requires_grad:
                        changed_trainable.append(name)
                    param.requires_grad = True

    if frozen_keywords:
        for name, param in model.named_parameters():
            if _match_keywords(name, frozen_keywords):
                if param.requires_grad:
                    changed_frozen.append(name)
                param.requires_grad = False

    if logger is not None:
        logger.info(
            f'[optimizer] requires_grad configured: +{len(changed_trainable)} trainable, '
            f'+{len(changed_frozen)} frozen after keyword filtering.'
        )
    return {
        'trainable': changed_trainable,
        'frozen': changed_frozen,
    }


def collect_optimizer_param_groups(model: torch.nn.Module, config: Any, logger: Optional[Any] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    set_trainable_by_keywords(model, config, logger=logger)

    base_lr = float(_get(config, 'lr', 1e-4))
    base_weight_decay = float(_get(config, 'weight_decay', 0.0))
    separate_no_decay = bool(_get(config, 'separate_no_decay', True))
    no_decay_keywords = _normalize_keywords(_get(config, 'no_decay_keywords', DEFAULT_NO_DECAY_KEYWORDS))
    if not no_decay_keywords:
        no_decay_keywords = list(DEFAULT_NO_DECAY_KEYWORDS)
    no_decay_weight_decay = float(_get(config, 'no_decay_weight_decay', 0.0))
    group_specs = _parse_param_group_specs(config)

    trainable_named_params: List[Tuple[str, torch.nn.Parameter]] = [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]
    if not trainable_named_params:
        raise ValueError('No trainable parameters found. Check your keyword filters or requires_grad settings.')

    assigned = set()
    group_buckets: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        'total_trainable_params': 0,
        'groups': [],
        'unassigned_names': [],
    }

    def make_bucket(name: str, lr: float, weight_decay: float) -> Dict[str, Any]:
        return {
            'name': name,
            'lr': float(lr),
            'weight_decay': float(weight_decay),
            'params': [],
            'param_names': [],
            'numel': 0,
        }

    # first-match-wins for explicit keyword groups
    for spec in group_specs:
        if not spec.get('trainable', True):
            continue
        bucket = make_bucket(
            name=str(spec.get('name', 'group')),
            lr=float(spec.get('lr', base_lr)),
            weight_decay=float(spec.get('weight_decay', base_weight_decay)),
        )
        include_keywords = spec.get('keywords', [])
        exclude_keywords = spec.get('exclude_keywords', [])
        match_all = bool(spec.get('match_all', False))

        for name, param in trainable_named_params:
            if name in assigned:
                continue
            if not _match_keywords(name, include_keywords, match_all=match_all):
                continue
            if _match_keywords(name, exclude_keywords):
                continue
            bucket['params'].append(param)
            bucket['param_names'].append(name)
            bucket['numel'] += param.numel()
            assigned.add(name)

        if bucket['params']:
            group_buckets.append(bucket)

    default_bucket = make_bucket('default', base_lr, base_weight_decay)
    for name, param in trainable_named_params:
        if name in assigned:
            continue
        default_bucket['params'].append(param)
        default_bucket['param_names'].append(name)
        default_bucket['numel'] += param.numel()
        summary['unassigned_names'].append(name)

    if default_bucket['params']:
        group_buckets.append(default_bucket)

    final_param_groups: List[Dict[str, Any]] = []
    for bucket in group_buckets:
        if not bucket['params']:
            continue

        if separate_no_decay:
            decay_params, no_decay_params = [], []
            decay_names, no_decay_names = [], []
            for name, param in zip(bucket['param_names'], bucket['params']):
                if param.ndim <= 1 or _contains_no_decay_keyword(name, no_decay_keywords):
                    no_decay_params.append(param)
                    no_decay_names.append(name)
                else:
                    decay_params.append(param)
                    decay_names.append(name)

            if decay_params:
                final_param_groups.append({
                    'params': decay_params,
                    'lr': bucket['lr'],
                    'weight_decay': bucket['weight_decay'],
                    'group_name': bucket['name'],
                    'group_variant': 'decay',
                })
                summary['groups'].append({
                    'name': bucket['name'],
                    'variant': 'decay',
                    'lr': bucket['lr'],
                    'weight_decay': bucket['weight_decay'],
                    'num_tensors': len(decay_params),
                    'numel': int(sum(p.numel() for p in decay_params)),
                    'sample_names': decay_names[:8],
                })
            if no_decay_params:
                final_param_groups.append({
                    'params': no_decay_params,
                    'lr': bucket['lr'],
                    'weight_decay': no_decay_weight_decay,
                    'group_name': bucket['name'],
                    'group_variant': 'no_decay',
                })
                summary['groups'].append({
                    'name': bucket['name'],
                    'variant': 'no_decay',
                    'lr': bucket['lr'],
                    'weight_decay': no_decay_weight_decay,
                    'num_tensors': len(no_decay_params),
                    'numel': int(sum(p.numel() for p in no_decay_params)),
                    'sample_names': no_decay_names[:8],
                })
        else:
            final_param_groups.append({
                'params': bucket['params'],
                'lr': bucket['lr'],
                'weight_decay': bucket['weight_decay'],
                'group_name': bucket['name'],
                'group_variant': 'all',
            })
            summary['groups'].append({
                'name': bucket['name'],
                'variant': 'all',
                'lr': bucket['lr'],
                'weight_decay': bucket['weight_decay'],
                'num_tensors': len(bucket['params']),
                'numel': int(bucket['numel']),
                'sample_names': bucket['param_names'][:8],
            })

    summary['total_trainable_params'] = int(sum(group['numel'] for group in summary['groups']))
    return final_param_groups, summary


def build_optimizer(model: torch.nn.Module, config: Any, logger: Optional[Any] = None) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
    param_groups, summary = collect_optimizer_param_groups(model, config, logger=logger)

    optimizer_name = str(_get(config, 'optimizer', 'AdamW')).lower()
    lr = float(_get(config, 'lr', 1e-4))
    weight_decay = float(_get(config, 'weight_decay', 0.0))
    betas = _parse_maybe_structured(_get(config, 'betas', (0.9, 0.999)))
    if isinstance(betas, list):
        betas = tuple(float(x) for x in betas)
    eps = float(_get(config, 'eps', 1e-8))
    momentum = float(_get(config, 'momentum', 0.9))
    nesterov = bool(_get(config, 'nesterov', False))
    alpha = float(_get(config, 'alpha', 0.99))
    centered = bool(_get(config, 'centered', False))

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(param_groups, lr=lr, weight_decay=weight_decay, momentum=momentum, alpha=alpha, eps=eps, centered=centered)
    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(param_groups, lr=lr, weight_decay=weight_decay, eps=eps)
    else:
        raise ValueError(f'Unsupported optimizer: {_get(config, "optimizer", None)}')

    if logger is not None:
        logger.info(
            f"[optimizer] Using {optimizer.__class__.__name__} with base lr={lr:.6g}, base wd={weight_decay:.6g}, "
            f"trainable_params={summary['total_trainable_params']:,}, groups={len(summary['groups'])}"
        )
        for idx, group in enumerate(summary['groups']):
            logger.info(
                f"[optimizer][group {idx}] {group['name']}::{group['variant']} | lr={group['lr']:.6g} | "
                f"wd={group['weight_decay']:.6g} | tensors={group['num_tensors']} | numel={group['numel']:,} | "
                f"sample={group['sample_names']}"
            )
    return optimizer, summary


def describe_trainable_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    total_params = 0
    trainable_params = 0
    trainable_names: List[str] = []
    frozen_names: List[str] = []
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    return {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'frozen_params': int(total_params - trainable_params),
        'trainable_names': trainable_names,
        'frozen_names': frozen_names,
    }


__all__ = [
    'build_optimizer',
    'collect_optimizer_param_groups',
    'set_trainable_by_keywords',
    'describe_trainable_parameters',
]