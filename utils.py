import numpy as np
import torch
import os
import h5py
import json
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

import IPython
e = IPython.embed

def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    max_episode_len = 0
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, "robot-franka", 'demo_{:04d}'.format(episode_idx))
        metadata_path = os.path.join(dataset_path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        ori_qpos = metadata.get('joint_qpos', [])
        qpos = np.array(ori_qpos[:-1])
        action = np.array(ori_qpos[1:]) # require slice off all elements' last element
        max_episode_len = max(max_episode_len, qpos.shape[0])
        # dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        # with h5py.File(dataset_path, 'r') as root:
        #     qpos = root['/observations/qpos'][()]
        #     qvel = root['/observations/qvel'][()]
        #     action = root['/action'][()]

        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.vstack(all_qpos_data)
    all_action_data = torch.vstack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos, "max_episode_len": max_episode_len}

    return stats


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def extract_model_state_dict(checkpoint):
    """Return model weights from supported checkpoint formats.

    Training checkpoints are wrapped under ``model``. Older ``policy_last``
    and ``policy_best`` files in this repository contain a raw state dict, so
    keep those files loadable for inference and weights-only initialization.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    if "model" in checkpoint:
        return checkpoint["model"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if checkpoint and all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in checkpoint.items()
    ):
        return checkpoint

    raise ValueError(
        "Checkpoint must contain 'model' or 'state_dict', or be a raw state dict"
    )

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def unnormalize_image(img_tensor, mean, std):
    img = img_tensor.clone().cpu().numpy()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.transpose(1, 2, 0)  # [H, W, C]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def normalize_action(action_tensor, normalizer):
        splits = [7, 21, 7, 21, 2]
        keys = [
            "/action/right_arm/joint_angle/rel",
            "/action/right_hand/joint_angle/rel",
            "/action/left_arm/joint_angle/rel",
            "/action/left_hand/joint_angle/rel",
            "/action/neck/joint_angle/rel",
        ]

        chunks = torch.split(action_tensor, splits, dim=-1)
        norm_chunks = []
        for key, chunk in zip(keys, chunks):
            norm_chunks.append(normalizer[key].normalize(chunk))
        return torch.cat(norm_chunks, dim=-1)

def denormalize_action(action_tensor, normalizer):
        splits = [7, 21, 7, 21, 2]
        keys = [
            "/action/right_arm/joint_angle/rel",
            "/action/right_hand/joint_angle/rel",
            "/action/left_arm/joint_angle/rel",
            "/action/left_hand/joint_angle/rel",
            "/action/neck/joint_angle/rel",
        ]

        chunks = torch.split(action_tensor, splits, dim=-1)
        denorm_chunks = []
        for key, chunk in zip(keys, chunks):
            denorm_chunks.append(normalizer[key].unnormalize(chunk))
        return torch.cat(denorm_chunks, dim=-1)

def normalize_obs_lowdim(lowdim_tensor, normalizer):
        splits = [7, 21, 7, 21, 2]
        keys = [
            "/state/right_arm/joint_angle",
            "/state/right_hand/joint_angle",
            "/state/left_arm/joint_angle",
            "/state/left_hand/joint_angle",
            "/state/neck/joint_angle",
        ]
        chunks = torch.split(lowdim_tensor, splits, dim=-1)
        norm_chunks = [
            normalizer[key].normalize(chunk)
            for key, chunk in zip(keys, chunks)
        ]
        return torch.cat(norm_chunks, dim=-1)

def denormalize_obs_lowdim(lowdim_tensor, normalizer):
        splits = [7, 21, 7, 21, 2]
        keys = [
            "/state/right_arm/joint_angle",
            "/state/right_hand/joint_angle",
            "/state/left_arm/joint_angle",
            "/state/left_hand/joint_angle",
            "/state/neck/joint_angle",
        ]
        chunks = torch.split(lowdim_tensor, splits, dim=-1)
        denorm_chunks = [
            normalizer[key].unnormalize(chunk)
            for key, chunk in zip(keys, chunks)
        ]
        return torch.cat(denorm_chunks, dim=-1)

def normalize_tactile(tactile_tensor, normalizer):
    return normalizer["/observe/tactile/total_force"].normalize(tactile_tensor)

def denormalize_tactile(tactile_tensor, normalizer):
    return normalizer["/observe/tactile/total_force"].unnormalize(tactile_tensor)

def normalize_tactile_next(tactile_tensor, normalizer):
    return normalizer["/observe/tactile/total_force/next"].normalize(tactile_tensor)

def denormalize_tactile_next(tactile_tensor, normalizer):
    return normalizer["/observe/tactile/total_force/next"].unnormalize(tactile_tensor)

def apply_joint_mask(qpos_tensor, mask, start_index):
    B, T, D = qpos_tensor.shape
    mask_tensor = torch.tensor(mask, dtype=qpos_tensor.dtype, device=qpos_tensor.device).view(1, 1, -1)
    qpos_tensor[..., start_index:start_index+len(mask)] *= mask_tensor
    return qpos_tensor
