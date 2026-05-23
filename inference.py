import pickle
import time
import cv2
from functools import wraps
import numpy as np
import torch
import zmq
import json
from PIL.ImageChops import offset

try:
    import pytorch_kinematics as pk
except:
    print("[unable init pytorch_kinematics] pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple pytorch-kinematics")

from torchvision.transforms.functional import resize

from utils import (
    normalize_obs_lowdim, normalize_tactile, denormalize_action, apply_joint_mask
)
from policy import ACTPolicy

from abc import ABC, abstractmethod

from dataset.pipelines_v2.compose import Compose
# env_pipeline + cfg imported conditionally in __main__ (tactile vs non-tactile pipeline)

import torchvision.utils as vutils
import os
from PIL import Image
import torchvision.transforms.functional as TF

import cv2

def unnormalize_image(img_tensor, mean, std):
    """
    Args:
        img_tensor: [3, H, W], float
        mean, std: list of 3 floats
    Returns:
        img: np.uint8, shape [H, W, 3]
    """
    img = img_tensor.clone().cpu().numpy()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.transpose(1, 2, 0)  # [H, W, C]
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img

def save_image_batch(image_tensor, save_dir="./vis_image", prefix="step", max_batch=1,
                     unnormalize=False, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Args:
        image_tensor: [B, N_cam, 3, H, W], normalized image
        unnormalize: whether to undo normalization
    """
    os.makedirs(save_dir, exist_ok=True)
    B, N_cam, C, H, W = image_tensor.shape
    for b in range(min(B, max_batch)):
        for cam in range(N_cam):
            img = image_tensor[b, cam]  # [3, H, W]
            if unnormalize:
                img_np = unnormalize_image(img, mean, std)  # → [H, W, 3], np.uint8
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_dir, f"{prefix}_b{b}_cam{cam}.jpg"), img_bgr)
            else:
                img_pil = TF.to_pil_image(img.cpu().detach())
                img_pil.save(os.path.join(save_dir, f"{prefix}_b{b}_cam{cam}.png"))


class Env(ABC):
    def __init__(self, action_dim=58, **kwargs):
        self.action_dim = action_dim
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:7778")
        self.obs = {}

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self):
        pass


class Ros2Env(Env):
    def reset(self, reset_meta=None):
        reset_info = {
            "action": np.zeros(self.action_dim).astype(float).tolist(),
            "reward": 0,
            "done": False,
            "info": {},
            "reset": True,
            "obs_nsteps": 1,
            "policy": "ACT",
        }
        if reset_meta is not None:
            reset_info.update(reset_meta)
        self.socket.send(json.dumps(reset_info).encode("utf-8"))
        print("Reseting...")
        self.obs = pickle.loads(self.socket.recv())

    def step(self, action=None, other_info=None):
        step_info = {
            "action": np.zeros(self.action_dim).astype(float).tolist(),
            "reward": 0,
            "done": False,
            "info": {},
            "reset": False,
            "obs_nsteps": 1,
            "policy": "ACT",
        }
        if other_info is not None:
            step_info.update(other_info)
        if action is not None:
            if isinstance(action, dict):
                step_info.update(action)
            else:
                step_info["action"] = np.array(action).astype(float).tolist()
        msg = json.dumps(step_info)
        self.socket.send(msg.encode("utf-8"))
        recv_bytes = self.socket.recv()
        self.obs = pickle.loads(recv_bytes)
        self.reward = self.obs["reward"]



def timing_measure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        if args[0].debug:
            print(f"Method '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

def tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().numpy().tolist()

# populated in __main__ from CLI args
config = {}


def build_config(args):
    return {
        'ckpt_path': args.ckpt_path,
        'stats_path': args.stats_path,
        'state_dim': 58,
        'policy_class': 'ACT',
        'use_tactile': args.use_tactile,
        'policy_config': {
            'num_queries': args.chunk_size,
            'hidden_dim': args.hidden_dim,
            'dim_feedforward': args.dim_feedforward,
            'kl_weight': 10,        # unused at inference (training loss only)
            'lr': 1e-4,             # unused at inference (no gradient steps)
            'lr_backbone': 1e-5,    # unused at inference
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': [
                "/observe/vision/head/stereo/lefteye/rgb",
                "/observe/vision/head/stereo/righteye/rgb",
                "/observe/vision/right_wrist/fisheye/rgb",
                "/observe/vision/left_wrist/fisheye/rgb",
            ],
            'use_tactile': args.use_tactile,
        },
    }


def postprocess_obs(raw_obs, device, use_tactile):
    obs_dict = {}
    camera_names = [
        "/observe/vision/head/stereo/lefteye/rgb",
        "/observe/vision/head/stereo/righteye/rgb",
        "/observe/vision/right_wrist/fisheye/rgb",
        "/observe/vision/left_wrist/fisheye/rgb"
    ]
    resize_shape = (224, 320)
    B = raw_obs[camera_names[0]].shape[0]
    cam_images = []

    for cam_name in camera_names:
            image = raw_obs[cam_name].to(device)  # shape: [B, 3, H, W]

            image = image.squeeze(0)  # [B, 3, H, W]

            resized_list = []

            for b in range(B):
                img_b = image[b]  # [3, H, W]
                resized = resize(img_b, resize_shape)  # → [3, 224, 320]
                resized_list.append(resized)

            resized_images = torch.stack(resized_list, dim=0)  # [B, 3, 224, 320]
            obs_dict[cam_name] = resized_images
            cam_images.append(resized_images)


    # stack all camera images -> [B, N_cam, 3, 224, 320]
    obs_dict["image"] = torch.stack(cam_images, dim=1)

    # joint state
    qpos_keys = [
        "/state/right_arm/joint_angle",
        "/state/right_hand/joint_angle",
        "/state/left_arm/joint_angle",
        "/state/left_hand/joint_angle",
        "/state/neck/joint_angle"
    ]

    qpos = torch.cat([raw_obs[k] for k in qpos_keys], dim=-1)  # [T, D]
    obs_dict["lowdim"] = qpos.to(device)

    # tactile
    if use_tactile:
        tac_key = "/observe/tactile/total_force"
        tactile = raw_obs[tac_key].to(device)  # [B, T, 120]
        obs_dict["tactile"] = tactile

    return obs_dict

def split_action(action):
    splits = [7, 21, 7, 21, 2]
    keys = [
        "/action/right_arm/joint_angle",
        "/action/right_hand/joint_angle",
        "/action/left_arm/joint_angle",
        "/action/left_hand/joint_angle",
        "/action/neck/joint_angle"
    ]
    chunks = torch.split(action, splits, dim=-1)
    result_dict = {k: v for k, v in zip(keys, chunks)}

    return result_dict


class InferencerV3:
    def __init__(self, model, cfg, env_pipe) -> None:
        """
        inference_type:
            - v1: blocking send, one action per step
            - v2: async send via buffer
        """
        class GenericObject:
            def __init__(self, data):
                self.__dict__.update(data)
        cfg = GenericObject(cfg)

        self.cfg = cfg
        self.env_pipe = env_pipe
        self.model = model
        norm_stats_cache = config["stats_path"]
        self.debug = cfg.env.get("debug", False)
        # self.env = build_env(cfg.env)
        self.env = Ros2Env(action_dim=cfg.env["action_dim"])
        self.dt = cfg.env["inference_dt"] # sample rate * collect dt
        self.vis_mode = cfg.env.get("vis_mode", False)
        self.custom_obs = cfg.env.get("custom_obs", None)

        # parse loader
        self.load_pipeline = cfg.env.pop("load_pipeline")
        self.load_keep = dict()
        self.load_cache = dict()
        self.load_custom_cache = dict()
        self.parse_loader()

        with open(norm_stats_cache, "rb") as f:
            self.stats = pickle.load(f)

        # build sim env
        self.temporal_agg = cfg.env.get("temporal_agg", None)
        self.inference_type = cfg.env.get("inference_type", "v1")
        self.action_output = cfg.env.get("action_output", None)
        self.chunk_size = cfg.env["num_queries"]
        self.execute = cfg.env.get("execute", None)

        # execute config
        self.obs_timestamp_key = cfg.env.get("obs_timestamp_key", None)
        self.action_execute_dt = cfg.env.get("action_execute_dt", None)

        # agg config
        self.agg_cfg = None
        if self.temporal_agg is not None and isinstance(self.temporal_agg, list):
            agg_cfg = dict()
            exp_weights_matrix_dict = dict() # {'action_n': [agg_num, agg_num, output_chunk], ...}
            for c_cfg in self.temporal_agg:
                action_n = c_cfg["action_name"]
                agg_flag = c_cfg.get("agg_flag", False)
                train_dt = c_cfg.get("train_dt", 0.1)
                offset = c_cfg.get("offset", 0)
                output_chunk = c_cfg.get("output_chunk", self.chunk_size)
                with_fk = c_cfg.get("with_fk", None)
                with_raw = c_cfg.get("with_raw", False)
                cfg_dict = dict(
                    agg_flag=agg_flag,
                    train_dt=train_dt,
                    offset=offset,
                    output_chunk=output_chunk,
                    with_fk=with_fk,
                    with_raw=with_raw,
                )
                if agg_flag:
                    hz_multiple = int(self.dt / train_dt) # agg offset
                    agg_num = len([i for i in range(0, self.chunk_size, hz_multiple)])
                    cfg_dict["hz_multiple"] = hz_multiple
                    cfg_dict["agg_num"] = agg_num
                    cfg_dict["action_list"] = []

                    exp_weights_matrix = np.zeros((agg_num, agg_num, output_chunk))
                    flag_matrix = np.zeros((agg_num, agg_num * hz_multiple + self.chunk_size))
                    for i in range(agg_num):
                        current_exp_weights_matrix = np.zeros((agg_num, output_chunk))
                        flag_matrix[- i - 1,
                        (agg_num - 1 - i) * hz_multiple: (agg_num - 1 - i) * hz_multiple + self.chunk_size] = 1
                        for j in range((agg_num - 1) * hz_multiple, (agg_num - 1) * hz_multiple + output_chunk, 1):
                            actions_populated = flag_matrix[:, j] != 0
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(sum(actions_populated)))
                            current_exp_weights_matrix[-len(exp_weights):,
                            j - (agg_num - 1) * hz_multiple] = exp_weights[
                                                               ::-1] / exp_weights.sum()
                        exp_weights_matrix[i] = current_exp_weights_matrix
                    exp_weights_matrix_dict[action_n] = torch.from_numpy(exp_weights_matrix)

                if with_fk is not None:
                    target_links = with_fk["target_links"]
                    hand_name_lookup = {jn: ji for ji, jn in enumerate(with_fk["proto_joint_names"])}
                    robot_chain = dict()
                    forward_jimap_dict = dict()
                    for target_link in target_links:
                        robot_chain[target_link] = pk.build_serial_chain_from_urdf(
                            open(with_fk["robot_urdf_path"]).read().encode("utf-8"), target_link
                        )
                        forward_jimap_dict[target_link] = [
                            hand_name_lookup[jn]
                            for jn in robot_chain[target_link].get_joint_parameter_names()
                        ]

                    cfg_dict["with_fk"] = dict(
                        target_links=target_links,
                        robot_chain=robot_chain,
                        forward_jimap_dict=forward_jimap_dict,
                    )

                agg_cfg[action_n] = cfg_dict
            self.agg_cfg = agg_cfg
            self.exp_weights_matrix_dict = exp_weights_matrix_dict

        # task code config
        self.task_code_cfg = cfg.env.get("task_code_cfg", None)

    def parse_loader(self):
        for loader in self.load_pipeline:
            if "Loader" not in loader['target']:
                continue
            load_key = loader['key']
            keep_length = len(np.arange(
                *loader['range'], loader['step']
            ))
            self.load_keep[load_key] = keep_length
        if self.custom_obs is not None:
            for key in self.custom_obs.keys():
                self.load_keep[key] = self.custom_obs[key]["keep_length"]

    @timing_measure
    def parse_obs_to_cache(self, obs):
        if obs.get("clear", False):
            print("[parse_obs] clear history cache!")
            self.load_cache = dict()

        for key in self.load_keep.keys():
            keep_length = self.load_keep[key]
            if key in obs:
                if key not in self.load_cache:
                    self.load_cache[key] = []
                input_data = np.array(obs[key])

                if self.vis_mode and "/images" in key:
                    cv2.imshow(
                        key,
                        cv2.cvtColor(
                            obs[key], cv2.COLOR_RGB2BGR
                        ),
                    )
                    cv2.waitKey(1)

                if self.vis_mode and "/deform" in key:
                    cv2.imshow(
                        key,
                        cv2.cvtColor(
                            obs[key], cv2.COLOR_RGB2BGR
                        ),
                    )
                    cv2.waitKey(1)

                if self.custom_obs is not None and key in self.custom_obs:
                    if self.custom_obs[key].get("choosen", None) is not None:
                        self.load_custom_cache[key] = [input_data[idx] for idx in self.custom_obs[key]["choosen"]]
                    else:
                        self.load_custom_cache[key] = input_data
                    continue

                if len(input_data.shape) == 2 and input_data.shape[0] == 1:
                    # assert input_data.shape[0] == 1, f"only support 1D data,"\
                    #                                  f"{key} with {input_data.shape}"
                    input_data = input_data[0]

                self.load_cache[key].append(input_data)
                while len(self.load_cache[key]) > keep_length:
                    self.load_cache[key].pop(0)
                if len(self.load_cache[key]) < keep_length:
                    insert_time = keep_length - len(self.load_cache[key])
                    insert_val = [self.load_cache[key][0] for _ in range(insert_time)]
                    self.load_cache[key] = insert_val + self.load_cache[key]

            elif "observation" in key:
                print(f"[parse_obs] {key} not in obs")
        return self.load_cache

    @timing_measure
    def model_forward(self, input_dict):
        with torch.inference_mode():
            net_results = self.model(**input_dict)
        return net_results

    @timing_measure
    def pipeline_for_input(self):
        input_dict = {k: np.array(v)
            for k, v in self.load_cache.items()}

        # if custom obs, add to input_dict and overwrite
        for key in self.load_custom_cache.keys():
            input_dict[key] = np.array(self.load_custom_cache[key])

        # add task code
        if self.task_code_cfg is not None:
            input_dict[self.task_code_cfg["output_name"]] = torch.from_numpy(
                np.array([self.task_code_cfg["value"]]).astype(int)
            ).long().cuda() # [B==1]
        elif "/task_code" in self.env.obs:
            input_dict['/task_code'] = torch.from_numpy(
                np.array([self.env.obs["/task_code"]]).astype(int)
            ).long().cuda()
        input_dict = self.env_pipe(input_dict)
        input_dict['norm_stats'] = self.stats


        return input_dict

    @timing_measure
    def step_and_getobs(self, result_dict):
        ts = self.env.step(result_dict)

    @timing_measure
    def action_aggregate(self, result_dict):
        def calcualte_fk_pos(action_ja, forward_jimap_dict, robot_chain, target_link):
            N = action_ja.shape[0]
            link_poses_urdf = robot_chain[target_link].forward_kinematics(action_ja[:, forward_jimap_dict[target_link]])
            link_mats_urdf = link_poses_urdf.get_matrix().reshape(N, 4, 4)
            pos = link_mats_urdf[..., :3, 3]
            return pos

        output_dict = dict()

        for idx, action_n in enumerate(self.action_output):
            if "rel" in action_n:
                assert self.agg_cfg is None or (not self.agg_cfg[action_n]["agg_flag"]), "rel action does not support agg_cfg"

            action_v = result_dict[action_n].squeeze(0)
            if self.agg_cfg is None or (not self.agg_cfg[action_n]["agg_flag"]):
                output_action = action_v
                with_raw = True if self.agg_cfg is None else self.agg_cfg[action_n]["with_raw"]
                output_chunk = 40 if self.agg_cfg is None else self.agg_cfg[action_n]["output_chunk"]
            else:
                agg_num = self.agg_cfg[action_n]["agg_num"]
                action_list = self.agg_cfg[action_n]["action_list"]
                hz_multiple = self.agg_cfg[action_n]["hz_multiple"]
                output_chunk = self.agg_cfg[action_n]["output_chunk"]
                with_raw = self.agg_cfg[action_n]["with_raw"]

                action_dim = action_v.shape[-1]
                num_queries = action_v.shape[0]

                action_list.append(action_v.detach().clone())
                pad_num = agg_num - len(action_list)

                if pad_num > 0:
                    pad_action = action_list[0]
                    action_list = [pad_action for _ in range(pad_num)] + action_list

                while len(action_list) > agg_num:
                    action_list.pop(0)

                # aggregation
                action_tensor = torch.zeros((agg_num, agg_num * hz_multiple + num_queries, action_dim)).to(action_v.device)
                for i in range(agg_num):
                    action_tensor[i, i * hz_multiple: (i * hz_multiple + num_queries), :] = action_list[i]

                exp_weights_matrix = self.exp_weights_matrix_dict[action_n]
                action_tensor = action_tensor[:, (agg_num - 1) * hz_multiple : (agg_num - 1) * hz_multiple + output_chunk] # [agg_num, output_chunk, action_dim]
                current_exp_weights = (
                    exp_weights_matrix[min(len(action_list), agg_num) - 1]
                    .unsqueeze(-1)
                    .expand(-1, -1, action_dim)
                    .to(action_v.device)
                )
                output_action = (action_tensor * current_exp_weights).sum(dim=0)
                assert output_action.shape == (output_chunk, action_dim), f"Expected output_action shape {(output_chunk, action_dim)}, but got {output_action.shape}"

            offset = 0 if self.agg_cfg is None else self.agg_cfg[action_n]["offset"]

            if self.inference_type == "v2":
                out_action = output_action[offset:,:].to(torch.float32)
                raw_action = action_v[offset:offset+output_chunk].to(torch.float32)
                output_dict[action_n] = tensor_to_list(out_action)
                if with_raw:
                    output_dict[f"{action_n}/raw"] = tensor_to_list(raw_action)

                with_fk = self.agg_cfg[action_n]["with_fk"]
                if with_fk is not None:
                    target_links = with_fk["target_links"]
                    robot_chain = with_fk["robot_chain"]
                    forward_jimap_dict = with_fk["forward_jimap_dict"]

                    for target_link in target_links:
                        if robot_chain[target_link].device != out_action.device:
                            robot_chain[target_link] = robot_chain[target_link].to(
                                dtype=out_action.dtype, device=out_action.device
                            )
                        out_pos = calcualte_fk_pos(out_action, forward_jimap_dict, robot_chain, target_link)
                        output_dict[f"{action_n}/{target_link}"] = tensor_to_list(out_pos)
                        if with_raw:
                            raw_pos = calcualte_fk_pos(raw_action, forward_jimap_dict, robot_chain, target_link)
                            output_dict[f"{action_n}/raw/{target_link}"] = tensor_to_list(raw_pos)
            else:
                raise NotImplementedError

        if self.obs_timestamp_key is not None:
            obs_input_timestamp = self.env.obs[self.obs_timestamp_key][-1]
            print(f"[obs_timestamp] obs {obs_input_timestamp:0.3f} at {time.time():0.6f}")
            output_dict['obs_timestamp'] = obs_input_timestamp
            output_dict['action_execute_ts'] = obs_input_timestamp + self.action_execute_dt

        return output_dict

    def run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        policy = ACTPolicy(config['policy_config'])
        checkpoint = torch.load(config['ckpt_path'], map_location=device)
        # print(checkpoint.keys())
        policy.load_state_dict(checkpoint['model'])
        # policy.load_state_dict(torch.load(config['ckpt_path'], map_location=device))
        policy.eval().to(device)
        print(f"loaded {config['ckpt_path']}")

        with open(config['stats_path'], 'rb') as f:
            normalizer = pickle.load(f)

        self.env.reset(reset_meta={
            "dt": self.dt,
            "base_name": "act",
            "env": self.cfg.env
        })

        with torch.inference_mode():

          self.query_count = 0
          while True:
                print(f"start {self.query_count}")
                start = time.time()
                obs = self.env.obs

                result_dict = None
                self.parse_obs_to_cache(obs)
                input_dict = self.pipeline_for_input()
                input_dict = postprocess_obs(input_dict, device, config['use_tactile'])

                qpos = normalize_obs_lowdim(input_dict["lowdim"], normalizer)

                # === apply masking to hand joint
                hand_mask = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]

                # right_hand starts at index 7 in qpos
                qpos = apply_joint_mask(qpos, hand_mask, start_index=7)

                # left_hand starts at index 7+21+7 = 35 in qpos
                qpos = apply_joint_mask(qpos, hand_mask, start_index=35)

                B, T, D = qpos.shape
                qpos = qpos.view(B, T * D)

                if config['use_tactile']:
                    tactile = normalize_tactile(input_dict["tactile"], normalizer)
                    B, T, D = tactile.shape
                    tactile = tactile.view(B, T * D)
                    tactile = tactile.to(device)

                qpos = qpos.to(device)

                save_image_batch(input_dict["image"], prefix="vis_epoch0", unnormalize=True)

                if config['use_tactile']:
                    a_hat = policy(qpos, input_dict["image"], tactile=tactile)
                else:
                    a_hat = policy(qpos, input_dict["image"], tactile=None)  # [B, T, D]
                action = denormalize_action(a_hat, normalizer).to(device)

                ref_qpos = input_dict["lowdim"][:, -1, :]  # [B, D]


                ref_qpos = ref_qpos.unsqueeze(1)         # [B, 1, D]

                abs_action = action + ref_qpos           # [B, T, D]


                result_dict = split_action(abs_action)

                result_dict = self.action_aggregate(result_dict)


                self.step_and_getobs(result_dict)

                duration = time.time() - start
                sleep_time = max(0, self.dt - duration)
                if obs.get("no_sleep", False):
                    sleep_time = 0.0
                print(f"duration: {duration:0.3f} / {self.dt:0.3f}", f"sleep: {sleep_time:0.3f}")
                time.sleep(sleep_time)

                self.query_count += 1


def sim_v3(model, cfg, env_pipe):
    infer = InferencerV3(model, cfg, env_pipe)
    infer.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to policy_*.ckpt')
    parser.add_argument('--stats_path', type=str, required=True, help='path to normalize.pkl')
    # must match training architecture
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--use_tactile', action='store_true')
    args = parser.parse_args()

    config.update(build_config(args))

    if args.use_tactile:
        from dataset.data_tactile import env_pipeline, cfg
    else:
        from dataset.data import env_pipeline, cfg

    env_pipe = Compose(env_pipeline)
    sim_v3(None, cfg, env_pipe)
