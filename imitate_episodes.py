import torch
import numpy as np
import os
import pickle
import argparse
import hashlib
import shutil
import matplotlib.pyplot as plt
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm
from einops import rearrange

from utils import compute_dict_mean, set_seed, detach_dict, extract_model_state_dict # helper functions
from utils import unnormalize_image, normalize_action, denormalize_action, normalize_obs_lowdim, denormalize_obs_lowdim, normalize_tactile, denormalize_tactile, normalize_tactile_next, denormalize_tactile_next, apply_joint_mask
from policy import ACTPolicy
# from visualize_episodes import save_videos
from dataset.ha_pipelinev2_dataset import HaPipelineV2DatasetD020
from dataset.data import data
from dataset.data_tactile import data_tactile
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm, trange

import IPython
e = IPython.embed

import torchvision.utils as vutils
import os
from PIL import Image
import torchvision.transforms.functional as TF

import cv2


CHECKPOINT_FORMAT_VERSION = 3
SUPPORTED_CHECKPOINT_FORMAT_VERSIONS = {2, CHECKPOINT_FORMAT_VERSION}
FULL_CHECKPOINT_KEYS = {
    'model',
    'optimizer',
    'scheduler',
    'epoch',
    'global_step',
    'min_val_loss',
}
NORMALIZATION_FILENAMES = {
    'dataset_stats': 'dataset_stats.pkl',
    'normalizer': 'normalize.pkl',
}


def _resolve_checkpoint_path(path, option_name):
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f'{option_name} checkpoint is not a readable file: {resolved}'
        )
    return resolved


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _normalization_metadata(run_dir):
    run_dir = Path(run_dir)
    metadata = {'version': 1}
    for key, filename in NORMALIZATION_FILENAMES.items():
        path = run_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing normalization file in run directory: {path}'
            )
        metadata[key] = {
            'filename': filename,
            'sha256': _sha256_file(path),
        }
    return metadata


def _validate_saved_normalization(checkpoint, run_dir):
    saved = checkpoint.get('normalization')
    format_version = checkpoint.get('format_version')
    if saved is None:
        if format_version == CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                'Checkpoint format 3 requires normalization metadata'
            )
        return
    if not isinstance(saved, dict) or saved.get('version') != 1:
        raise ValueError('Unsupported checkpoint normalization metadata')

    actual = _normalization_metadata(run_dir)
    for key, filename in NORMALIZATION_FILENAMES.items():
        entry = saved.get(key)
        if not isinstance(entry, dict):
            raise ValueError(
                f'Checkpoint normalization metadata is missing {key}'
            )
        if entry.get('filename') != filename:
            raise ValueError(
                f'Checkpoint normalization filename mismatch for {key}'
            )
        if entry.get('sha256') != actual[key]['sha256']:
            raise ValueError(
                f'Checkpoint normalization hash mismatch for {filename}'
            )


def _validate_full_checkpoint(
    checkpoint,
    policy_config,
    future_tactile_curriculum,
):
    if not isinstance(checkpoint, dict):
        raise TypeError('A full resume checkpoint must be a dictionary')

    missing = sorted(FULL_CHECKPOINT_KEYS - checkpoint.keys())
    if missing:
        raise ValueError(
            'The file passed to --resume_path contains partial training state '
            f'and cannot be resumed; missing: {", ".join(missing)}.'
        )

    if (
        isinstance(checkpoint['epoch'], bool)
        or not isinstance(checkpoint['epoch'], int)
        or checkpoint['epoch'] < 0
    ):
        raise ValueError('Resume checkpoint epoch must be a non-negative integer')
    if (
        isinstance(checkpoint['global_step'], bool)
        or not isinstance(checkpoint['global_step'], int)
        or checkpoint['global_step'] < 0
    ):
        raise ValueError(
            'Resume checkpoint global_step must be a non-negative integer'
        )
    for key in ('model', 'optimizer', 'scheduler'):
        if not isinstance(checkpoint[key], dict):
            raise TypeError(f'Resume checkpoint {key} must be a dictionary')

    if 'last_epoch' not in checkpoint['scheduler']:
        raise ValueError(
            'Resume checkpoint scheduler is missing last_epoch'
        )
    scheduler_last_epoch = checkpoint['scheduler']['last_epoch']
    if (
        isinstance(scheduler_last_epoch, bool)
        or not isinstance(scheduler_last_epoch, int)
        or scheduler_last_epoch < 0
    ):
        raise ValueError(
            'Resume checkpoint scheduler.last_epoch must be a non-negative '
            'integer'
        )
    if scheduler_last_epoch != checkpoint['global_step']:
        raise ValueError(
            'Resume checkpoint scheduler.last_epoch does not match global_step'
        )

    format_version = checkpoint.get('format_version')
    if format_version is not None:
        if format_version not in SUPPORTED_CHECKPOINT_FORMAT_VERSIONS:
            raise ValueError(
                f'Unsupported checkpoint format_version: {format_version}'
            )
        if 'policy_config' not in checkpoint:
            raise ValueError(
                'Versioned resume checkpoint is missing policy_config'
            )

    saved_policy_config = checkpoint.get('policy_config')
    if (
        saved_policy_config is not None
        and saved_policy_config != policy_config
    ):
        raise ValueError(
            'Resume checkpoint policy_config does not match the current '
            f'configuration. Saved: {saved_policy_config}; current: '
            f'{policy_config}'
        )

    # Legacy checkpoints may not contain this field. In that case, use the
    # current CLI setting (default: 75) without changing legacy behavior.
    saved_curriculum = checkpoint.get('future_tactile_curriculum')
    if (
        saved_curriculum is not None
        and saved_curriculum != future_tactile_curriculum
    ):
        raise ValueError(
            'Resume checkpoint future tactile curriculum does not match the '
            'current training configuration'
        )

    if format_version == CHECKPOINT_FORMAT_VERSION:
        scheduler_config = checkpoint.get('scheduler_config')
        if not isinstance(scheduler_config, dict):
            raise ValueError(
                'Checkpoint format 3 requires scheduler_config metadata'
            )


def _prepare_training_checkpoint(
    resume_path,
    policy_config,
    future_tactile_curriculum,
):
    resume_path = _resolve_checkpoint_path(resume_path, '--resume_path')
    if resume_path is None:
        return None

    checkpoint = torch.load(resume_path, map_location='cpu')
    is_full_candidate = any(
        key in checkpoint for key in ('optimizer', 'scheduler')
    ) if isinstance(checkpoint, dict) else False
    if is_full_candidate:
        _validate_full_checkpoint(
            checkpoint,
            policy_config,
            future_tactile_curriculum,
        )
        source_run_dir = resume_path.parent
        source_statistics = {
            key: source_run_dir / filename
            for key, filename in NORMALIZATION_FILENAMES.items()
        }
        for path in source_statistics.values():
            if not path.is_file():
                raise FileNotFoundError(
                    'Full resume requires checkpoint-sibling statistics: '
                    f'{path}'
                )
        _validate_saved_normalization(checkpoint, source_run_dir)
        return {
            'mode': 'resume',
            'path': resume_path,
            'checkpoint': checkpoint,
            'source_statistics': source_statistics,
        }

    # Raw and model-only checkpoints keep the original --resume_path behavior:
    # load the model weights and start a fresh optimizer/scheduler at epoch 0.
    extract_model_state_dict(checkpoint)
    return {
        'mode': 'weights',
        'path': resume_path,
        'checkpoint': checkpoint,
    }



def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_root = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    use_tactile = args['use_tactile']
    resume_path = args['resume_path']
    tactile_teacher_forcing_epochs = args['tactile_teacher_forcing_epochs']
    if tactile_teacher_forcing_epochs < 0:
        raise ValueError('--tactile_teacher_forcing_epochs must be non-negative')

    episode_len = 10000
    camera_names = ['/observe/vision/head/stereo/lefteye/rgb','/observe/vision/head/stereo/righteye/rgb','/observe/vision/right_wrist/fisheye/rgb','/observe/vision/left_wrist/fisheye/rgb']

    # fixed parameters
    state_dim = 58
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'use_tactile': use_tactile
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    future_tactile_curriculum = {
        'version': 1,
        'mode': 'hard_switch',
        'teacher_forcing_epochs': tactile_teacher_forcing_epochs,
    } if use_tactile else None

    checkpoint_info = _prepare_training_checkpoint(
        resume_path,
        policy_config,
        future_tactile_curriculum,
    )

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ckpt_dir = os.path.join(ckpt_root, timestamp)

    if use_tactile:
        ckpt_dir = ckpt_dir + "_tactile"
        timestamp = timestamp + "_tactile"

    os.makedirs(ckpt_dir, exist_ok=True)

    if checkpoint_info is not None and checkpoint_info['mode'] == 'resume':
        for key, source_path in checkpoint_info['source_statistics'].items():
            destination = Path(ckpt_dir) / NORMALIZATION_FILENAMES[key]
            if source_path.resolve() != destination.resolve():
                shutil.copy2(source_path, destination)

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        # 'real_robot': not is_sim,
        'use_tactile': use_tactile,
        'future_tactile_curriculum': future_tactile_curriculum,
        'lr_config': {
            'policy': 'CosineAnnealing',
            'warmup': 'linear',
            'warmup_iters': 1000,
            'warmup_ratio': 1.0 / 10,
            'min_lr_ratio': 1e-1,
        }
    }

    norm_stats_cache = os.path.join(ckpt_dir, 'dataset_stats.pkl')

    data['train']['norm_stats_cache'] = norm_stats_cache
    data['val']['norm_stats_cache'] = norm_stats_cache
    data_tactile['train']['norm_stats_cache'] = norm_stats_cache
    data_tactile['val']['norm_stats_cache'] = norm_stats_cache


    if not use_tactile:
        train_dataset = HaPipelineV2DatasetD020(**data['train'])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)

        # val_dataset = HaPipelineV2DatasetD020(**data['val'])
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)
    else:
        train_dataset = HaPipelineV2DatasetD020(**data_tactile['train'])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)

        # val_dataset = HaPipelineV2DatasetD020(**data_tactile['val'])
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=False, num_workers=36, prefetch_factor=1)

    stats_path = os.path.join(ckpt_dir, f'normalize.pkl')
    if checkpoint_info is not None and checkpoint_info['mode'] == 'resume':
        with open(stats_path, 'rb') as f:
            normalizer = pickle.load(f)
    else:
        normalizer = train_dataset.get_normalizer()
        with open(stats_path, 'wb') as f:
            pickle.dump(normalizer, f)

    config['normalization'] = _normalization_metadata(ckpt_dir)

    selected_epoch, selected_metric, selection = train_bc(
        train_dataloader,
        normalizer,
        train_dataset,
        timestamp,
        config,
        checkpoint_info=checkpoint_info,
    )
    print(
        f'Checkpoint selection: {selection}, metric {selected_metric:.6f} '
        f'@ epoch{selected_epoch}'
    )


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def forward_pass(
    data,
    policy,
    normalizer,
    device,
    use_tactile,
    use_gt_tactile=None,
):
    image_data = data["image"]               # [B, N_cam, 3, H, W]
    qpos_data = data["lowdim"]               # [B, T1, D1]
    action_data = data["action"]            # [B, T, D_action]
    is_pad = data["action_mask"]            # [B, T]

    # normalize
    qpos_data_norm = normalize_obs_lowdim(qpos_data, normalizer)  # [B, T1, D1]
    action_data_norm = normalize_action(action_data, normalizer)  # [B, T, D_action]

    # === apply masking to hand joint
    hand_mask = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    # right_hand mask
    qpos_data_norm = apply_joint_mask(qpos_data_norm, hand_mask, start_index=7)

    # left_hand mask
    qpos_data_norm = apply_joint_mask(qpos_data_norm, hand_mask, start_index=35)

    # flatten
    B, T1, D1 = qpos_data_norm.shape
    qpos_data_norm = qpos_data_norm.view(B, T1 * D1)  # → [B, T1 * D1]

    # move to device
    qpos_data_norm = qpos_data_norm.to(device)
    image_data = image_data.to(device)
    action_data_norm = action_data_norm.to(device)
    is_pad = is_pad.to(device)

    if use_tactile:
        if use_gt_tactile is None:
            raise ValueError(
                'use_gt_tactile must be specified for tactile training/evaluation'
            )
        tactile = data["tactile"]                          # [B, T2, D2]
        tactile_norm = normalize_tactile(tactile, normalizer)  # normalize
        B, T2, D2 = tactile_norm.shape
        tactile_norm = tactile_norm.view(B, T2 * D2)                # → [B, T2 * D2]
        tactile_norm = tactile_norm.to(device)                     

        tactile_next = data["tactile_next"]                          # [B, T2, D2]
        tactile_next_norm = normalize_tactile_next(tactile_next, normalizer)  # normalize
        tactile_next_norm = tactile_next_norm.to(device)                     
        return policy(
            qpos_data_norm,
            image_data,
            action_data_norm,
            is_pad,
            device,
            tactile_norm,
            tactile_next_norm,
            use_gt_tactile=use_gt_tactile,
        )

    return policy(qpos_data_norm, image_data, action_data_norm, is_pad, device)


def train_bc(
    train_dataloader,
    normalizer,
    dataset,
    timestamp,
    config,
    checkpoint_info=None,
):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    use_tactile = config['use_tactile']
    future_tactile_curriculum = config.get('future_tactile_curriculum')
    normalization = config['normalization']
    if use_tactile:
        if future_tactile_curriculum is None:
            raise ValueError('Missing future_tactile_curriculum for tactile training')
        if future_tactile_curriculum.get('version') != 1:
            raise ValueError('Unsupported future tactile curriculum version')
        if future_tactile_curriculum.get('mode') != 'hard_switch':
            raise ValueError(
                'Only future tactile curriculum mode="hard_switch" is supported'
            )
        teacher_forcing_epochs = future_tactile_curriculum.get(
            'teacher_forcing_epochs'
        )
        if not isinstance(teacher_forcing_epochs, int) or teacher_forcing_epochs < 0:
            raise ValueError(
                'future_tactile_curriculum.teacher_forcing_epochs '
                'must be a non-negative integer'
            )
        print(
            'Future tactile curriculum: ground truth for epochs '
            f'[0, {teacher_forcing_epochs}), predicted from epoch '
            f'{teacher_forcing_epochs}'
        )
    else:
        teacher_forcing_epochs = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(seed)

    start_epoch = 0
    global_step = 0
    min_val_loss = np.inf
    best_ckpt_info = None

    from transformers import get_cosine_schedule_with_warmup

    policy = make_policy(policy_class, policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_class, policy)

    # === 构建 scheduler ===
    total_iters = num_epochs * len(train_dataloader)
    scheduler_config = {
        'name': 'cosine_schedule_with_warmup',
        'num_warmup_steps': config['lr_config']['warmup_iters'],
        'num_training_steps': total_iters,
        'steps_per_epoch': len(train_dataloader),
    }

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=scheduler_config['num_warmup_steps'],
        num_training_steps=scheduler_config['num_training_steps'],
    )

    validation_history = []
    global_step = 0
    min_val_loss = np.inf
    best_ckpt_info = None

    if checkpoint_info is not None:
        checkpoint = checkpoint_info.pop('checkpoint')
        checkpoint_mode = checkpoint_info['mode']
        checkpoint_path = checkpoint_info['path']
        if checkpoint_mode == 'resume':
            saved_scheduler_config = checkpoint.get('scheduler_config')
            if (
                saved_scheduler_config is not None
                and saved_scheduler_config != scheduler_config
            ):
                raise ValueError(
                    'Resume checkpoint scheduler_config does not match the '
                    f'current training schedule. Saved: '
                    f'{saved_scheduler_config}; current: {scheduler_config}'
                )

        policy.load_state_dict(extract_model_state_dict(checkpoint))
        if checkpoint_mode == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            min_val_loss = checkpoint['min_val_loss']
            best_ckpt_info = checkpoint.get('best_ckpt_info', None)
            print(
                f'[Resume] Full checkpoint {checkpoint_path}; continuing '
                f'from epoch {start_epoch}, global step {global_step}'
            )
        else:
            print(
                f'[Resume] Model-only checkpoint {checkpoint_path}; starting '
                'from epoch 0 with a fresh optimizer and scheduler'
            )
        del checkpoint

    last_epoch = start_epoch - 1
    last_train_loss = None

    for epoch in tqdm(range(start_epoch, num_epochs)):
        step_log = {}
        print(f'\nEpoch {epoch}')
        use_gt_tactile = use_tactile and epoch < teacher_forcing_epochs
        if use_tactile:
            tactile_source = 'ground truth' if use_gt_tactile else 'predicted'
            print(f'Future tactile input for action decoder: {tactile_source}')
        # if epoch % 5 == 0:
        #     # validation
        #     # with torch.inference_mode():
        #     with torch.no_grad():
        #         policy.eval()
        #         epoch_dicts = []
        #         for data in tqdm(val_dataloader, desc="Validation", leave=False):
        #             data = dataset.postprocess(data, device, use_tactile)
        #             forward_dict = forward_pass(
        #                 data, policy, normalizer, device, use_tactile,
        #                 use_gt_tactile=False,
        #             )
        #             epoch_dicts.append(forward_dict)

        #         epoch_summary = compute_dict_mean(epoch_dicts)
        #         validation_history.append(epoch_summary)

        #         epoch_val_loss = epoch_summary['loss']
        #         if epoch_val_loss < min_val_loss:
        #             min_val_loss = epoch_val_loss
        #             best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

        #     print(f'Val loss:   {epoch_val_loss:.5f}')
        #     summary_string = ''
        #     for k, v in epoch_summary.items():
        #         summary_string += f'{k}: {v.item():.3f} '
        #     print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        epoch_history = []
        with tqdm(train_dataloader, desc=f"Train Epoch {epoch}", leave=False) as tepoch:
            # for batch_idx, data in enumerate(train_dataloader):
            for batch_idx, data in enumerate(tepoch):
                data = dataset.postprocess(data, device, use_tactile)
                forward_dict = forward_pass(
                    data,
                    policy,
                    normalizer,
                    device,
                    use_tactile,
                    use_gt_tactile=use_gt_tactile,
                )
                # backward
                loss = forward_dict['loss']
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_history.append(detach_dict(forward_dict))

                tepoch.set_postfix(
                    loss=loss.item(),
                    refresh=False
                )

                global_step += 1

        if not epoch_history:
            raise RuntimeError("Training dataloader produced no batches")
        epoch_summary = compute_dict_mean(epoch_history)
        epoch_train_loss = epoch_summary['loss']
        last_epoch = epoch
        last_train_loss = float(epoch_train_loss.detach().cpu())
        print(f'Train loss: {epoch_train_loss:.5f}')

        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 5 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_loss_{epoch_train_loss:.3f}.ckpt')
            # torch.save(policy.state_dict(), ckpt_path)
            torch.save({
                'format_version': CHECKPOINT_FORMAT_VERSION,
                'model': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scheduler_config': scheduler_config,
                'epoch': epoch,
                'global_step': global_step,
                'min_val_loss': min_val_loss,
                'policy_config': policy_config,
                'future_tactile_curriculum': future_tactile_curriculum,
                'normalization': normalization,
                'selection': 'periodic',
            }, ckpt_path)

    if last_train_loss is None:
        raise RuntimeError(
            f"No epoch ran: start_epoch={start_epoch}, num_epochs={num_epochs}"
        )

    last_checkpoint = {
        'format_version': CHECKPOINT_FORMAT_VERSION,
        'model': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scheduler_config': scheduler_config,
        'epoch': last_epoch,
        'global_step': global_step,
        'min_val_loss': min_val_loss,
        'policy_config': policy_config,
        'future_tactile_curriculum': future_tactile_curriculum,
        'normalization': normalization,
        'metric_name': 'train_loss',
        'metric': last_train_loss,
        'selection': 'last',
    }
    last_ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    torch.save(last_checkpoint, last_ckpt_path)

    if best_ckpt_info is None:
        selected_epoch = last_epoch
        selected_metric = last_train_loss
        selection = 'last_no_validation'
        best_checkpoint = dict(last_checkpoint)
        best_checkpoint['selection'] = selection
    else:
        selected_epoch, selected_metric, selected_state_dict = best_ckpt_info
        selected_metric = float(selected_metric)
        selection = 'validation'
        best_checkpoint = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'model': selected_state_dict,
            'policy_config': policy_config,
            'future_tactile_curriculum': future_tactile_curriculum,
            'normalization': normalization,
            'epoch': selected_epoch,
            'metric_name': 'val_loss',
            'metric': selected_metric,
            'selection': selection,
        }
    best_ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    torch.save(best_checkpoint, best_ckpt_path)
    print(
        f'Training finished:\nSeed {seed}, {selection} metric '
        f'{selected_metric:.6f} at epoch {selected_epoch}'
    )

    return selected_epoch, selected_metric, selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_tactile', action='store_true')
    parser.add_argument(
        '--tactile_teacher_forcing_epochs',
        type=int,
        default=75,
        help=(
            'Number of initial epochs that feed ground-truth future tactile '
            'to the action decoder; predicted tactile is used afterwards'
        ),
    )
    parser.add_argument(
        '--resume_path',
        type=str,
        default=None,
        help=(
            'Path to a checkpoint. A full checkpoint restores the complete '
            'training state; a raw or model-only checkpoint loads weights '
            'and starts from epoch 0.'
        ),
    )

    main(vars(parser.parse_args()))
