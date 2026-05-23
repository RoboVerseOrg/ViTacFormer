"""params settings."""
# train params
import torch
import socket
import copy
import os
import numpy as np

total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1000)
if total_memory < 16:
    batch_size = 2
    workers = 1
elif 80<total_memory<90:
    batch_size = 80
    workers = 12
elif 90<total_memory<100:
    batch_size = 93
    workers = 18

batch_size = 256
workers = 16

pin_memory = False
num_gpu = torch.cuda.device_count()  # NOTE: CUDA_VISIBLE_DEVICES

max_iters = 100000
save_interval = 5000
val_interval = max_iters


work_dir = "work_dirs/"

"""
data settings
"""
TMODE1 = [
    "./example_data"
]

DATA_TYPE="0.3.0"
norm_stats_cache=None
inference_dt = 0.1
collect_dt = 0.033

dataset_config = dict(
    dataset_type="HaPipelineV2DatasetD020",
    seed=0,
    data_root=TMODE1,
    # seq_list
    train_skip=None,
    test_skip=None,
    train_seq_list=None,
    test_seq_list=None,
    # obs information
    train_ratio=1.0,
    process_range_desc=[
        "/anno/valid/step_table",
    ], 
    sample_rate=3,
    tac_sample_rate=1,
    collect_dt=0.033,
)

camera_preprcess_info=[
    dict(
        name='/observe/vision/head/stereo/lefteye/rgb',
        img_size=[180, 320],
    ),
    dict(
        name='/observe/vision/head/stereo/righteye/rgb',
        img_size=[180, 320],
    ),
    dict(
        name='/observe/vision/right_wrist/fisheye/rgb', # 原图倒转
        flip="xy",
        crop=[27, 103, 552, 660],
        img_size=[256, 280],
    ),
    dict(
        name='/observe/vision/left_wrist/fisheye/rgb', # 原图倒转
        flip="xy",
        crop=[27, 103, 552, 660],
        img_size=[256, 280],
    ),
]
camera_names = list(c["name"] for c in camera_preprcess_info)

h5_lowdim_input = [
    "/state/right_arm/joint_angle",
    "/state/right_hand/joint_angle",
    "/state/left_arm/joint_angle",
    "/state/left_hand/joint_angle",
    "/state/neck/joint_angle",
]
h5_action_input = [
    "/action/right_arm/joint_angle",
    "/action/right_hand/joint_angle",
    "/action/left_arm/joint_angle",
    "/action/left_hand/joint_angle",
    "/action/neck/joint_angle",
]

lowdim_to_model = [
    "/state/right_arm/joint_angle",
    "/state/right_hand/joint_angle",
    "/state/left_arm/joint_angle",
    "/state/left_hand/joint_angle",
    "/state/neck/joint_angle",
]
action_to_model = [
    "/action/right_arm/joint_angle/rel",
    "/action/right_hand/joint_angle/rel",
    "/action/left_arm/joint_angle/rel",
    "/action/left_hand/joint_angle/rel",
    "/action/neck/joint_angle/rel",
]

h5_tactile_input = [
    "/observe/tactile/right_thumb/force",
    "/observe/tactile/right_thumb/torque",
    "/observe/tactile/right_index/force",
    "/observe/tactile/right_index/torque",
    "/observe/tactile/right_middle/force",
    "/observe/tactile/right_middle/torque",
    "/observe/tactile/right_ring/force",
    "/observe/tactile/right_ring/torque",
    "/observe/tactile/right_little/force",
    "/observe/tactile/right_little/torque",
    "/observe/tactile/left_thumb/force",
    "/observe/tactile/left_thumb/torque",
    "/observe/tactile/left_index/force",
    "/observe/tactile/left_index/torque",
    "/observe/tactile/left_middle/force",
    "/observe/tactile/left_middle/torque",
    "/observe/tactile/left_ring/force",
    "/observe/tactile/left_ring/torque",
    "/observe/tactile/left_little/force",
    "/observe/tactile/left_little/torque",
]
tacforce_name = "/observe/tactile/total_force"
model_tac_state_dim_list = [3 for _ in range(len(h5_tactile_input))]
model_tac_state_dim = sum(model_tac_state_dim_list)

lowdim_to_normalize = lowdim_to_model + action_to_model + [tacforce_name] + [f"{tacforce_name}/next"]

history_cnt = 6    
chunk_size = 100
tac_history_cnt = int(history_cnt*dataset_config["sample_rate"]/float(dataset_config["tac_sample_rate"]))  # 3*6/1=18

"""
save dir
"""
load_from = None

'''
input and pipeline
'''
lowdim_load_pipeline = [
    # state loader
    *[dict(
        target="dataset.pipelines_v2.slice_loader.SliceLoader",
        key=name,
        range=[0, chunk_size],
        step=1,
        padding_value="closest",
        hf_conf=dict(
            index_key="aligned_index",
            use_index_value=False,
        ),
        dtype=np.float32,
    ) for name in h5_action_input],
    *[dict(
        target="dataset.pipelines_v2.slice_loader.SliceLoader",
        key=name,
        range=[-(history_cnt - 1) * dataset_config["sample_rate"], 1],
        step=dataset_config["sample_rate"],
        padding_value="closest",
        hf_conf=dict(
            index_key="aligned_index",
            use_index_value=True,   # NOTE: 低频history输入
        ),
        dtype=np.float32,
    ) for name in h5_lowdim_input],
    *[dict(
        target="dataset.pipelines_v2.slice_loader.SliceLoader",
        key=name,
        range=[-(tac_history_cnt - 1) * dataset_config["tac_sample_rate"], 1],
        step=dataset_config["tac_sample_rate"],  # NOTE: 高频30Hz触觉信号输入
        padding_value="closest",
        hf_conf=None,
        dtype='float32',
    ) for name in h5_tactile_input],
    *[dict(
        target="dataset.pipelines_v2.slice_loader.SliceLoader",
        key=name,
        out_key = f"{name}/next",
        range=[0, (tac_history_cnt) * dataset_config["tac_sample_rate"]],
        step=dataset_config["tac_sample_rate"],  # NOTE: 高频30Hz触觉信号输入
        padding_value="closest",
        hf_conf=None,
        dtype='float32',
    ) for name in h5_tactile_input],
]

endpoint_load_pipeline = [
    *[dict(
        target="dataset.pipelines_v2.slice_loader.EndpointLoader",
        key=camera_names[0],
        range=[0, chunk_size],
        end_length=chunk_size,
        dtype=np.float32,
    )],
]

img_load_pipeline = [
    *[dict(
        target="dataset.pipelines_v2.slice_loader.ImageLoader",
        key=name,
        range=[0, 1],
        step=dataset_config["sample_rate"],
        padding_value="closest",
    ) for name in camera_names],
]


lowdim_process_pipeline = [
    *[dict(
        target="dataset.pipelines_v2.transform.RelativeGenerate",
        key=name,
        ref_key=h5_lowdim_input[i],
        meta={"type": "rel_abs"},
    ) for i, name in enumerate(h5_action_input)],
    dict(
        target="dataset.pipelines_v2.transform.ConcatData",
        keys=h5_tactile_input,
        output_key=tacforce_name,  # [L, C]
        dim=-1,
    ),
    dict(
        target="dataset.pipelines_v2.transform.RelativeGenerate",
        key=tacforce_name,
        ref_key=tacforce_name,
        ref_index=0,
        meta={"type": "rel_abs"},
    ),
    dict(
        target="dataset.pipelines_v2.transform.ConcatData",
        keys=[tacforce_name, f"{tacforce_name}/rel"],
        output_key=tacforce_name,  # [L, 2C], abs+rel
        dim=-1,
    ),
    dict(
        target="dataset.pipelines_v2.transform.ConcatData",
        keys=[f"{name}/next" for name in h5_tactile_input],
        output_key=f"{tacforce_name}/next",  # [L, C]
        dim=-1,
    ),
    dict(
        target="dataset.pipelines_v2.transform.RelativeGenerate",
        key=f"{tacforce_name}/next",
        ref_key=f"{tacforce_name}/next",
        ref_index=0,
        meta={"type": "rel_abs"},
    ),
    dict(
        target="dataset.pipelines_v2.transform.ConcatData",
        keys=[f"{tacforce_name}/next", f"{tacforce_name}/next/rel"],
        output_key=f"{tacforce_name}/next",  # [L, 2C], abs+rel
        dim=-1,
    ),
]

img_process_pipeline = [
    ### image processor
    *[dict(
        target="dataset.pipelines_v2.transform.ImageProcess",
        key=cam_info["name"],
        norm=True,
        aug=True,
        aug_geo=True,
        aug_color=True,
        camera_preprcess_info=cam_info,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        aug_conf=dict(
            crop_ratio=0.98,
            rotate_range=[-0.3, 0.3],
            brightness_delta=20,
            contrast_range=(0.8, 1.2),
            saturation_range=(0.8, 1.2)
        ),
        debug=False,
    ) for cam_info in camera_preprcess_info],
]

train_load_pipeline = lowdim_load_pipeline + img_load_pipeline
test_load_pipeline = lowdim_load_pipeline + img_load_pipeline
process_pipeline = lowdim_process_pipeline + img_process_pipeline
stat_pipeline = lowdim_load_pipeline + lowdim_process_pipeline
train_pipeline = train_load_pipeline + process_pipeline

test_pipeline = []
for pipe in copy.deepcopy(train_pipeline):
    if pipe["target"] == "dataset.pipelines_v2.transform.ImageProcess":
        pipe["aug"] = False
    test_pipeline.append(pipe)

data_tactile = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=workers,
    train=dict(
        data_type=DATA_TYPE,
        type=dataset_config["dataset_type"],
        seed=dataset_config["seed"],
        dataset_config=dataset_config,
        skip_mirrored_data=False,
        test_mode=False,
        norm_stats_cache=norm_stats_cache,
        pipeline=train_pipeline,
        stat_pipeline=stat_pipeline,
        stat_sample_step=10,
        stat_worker=16,
        normalize_keys=lowdim_to_normalize,
        skip=dataset_config.get("train_skip", None),
        seq_list=dataset_config.get("train_seq_list", None),
        length_keys=h5_action_input,
        work_dir=work_dir,
    ),
    val=dict(
        data_type=DATA_TYPE,
        limit_num=batch_size * 20,
        type=dataset_config["dataset_type"],
        seed=dataset_config["seed"],
        dataset_config=dataset_config,
        skip_mirrored_data=False,
        test_mode=True,
        norm_stats_cache=norm_stats_cache,
        pipeline=test_pipeline,
        normalize_keys=lowdim_to_normalize,
        skip=dataset_config.get("test_skip", None),
        seq_list=dataset_config.get("test_seq_list", None),
        length_keys=h5_action_input,
        work_dir=work_dir,
    ),
    test=dict(
        data_type=DATA_TYPE,
        limit_num=batch_size * 20,
        type=dataset_config["dataset_type"],
        seed=dataset_config["seed"],
        dataset_config=dataset_config,
        skip_mirrored_data=False,
        test_mode=True,
        norm_stats_cache=norm_stats_cache,
        pipeline=test_pipeline,
        normalize_keys=lowdim_to_normalize,
        skip=dataset_config.get("test_skip", None),
        seq_list=dataset_config.get("test_seq_list", None),
        length_keys=h5_action_input,
        work_dir=work_dir,
    ),
)


"""
inference config
"""
env_pipeline = [
    *copy.deepcopy(process_pipeline),
    dict(
        target="dataset.pipelines_v2.transform.EnvAddBatchDim",
        keys=set(camera_names + lowdim_to_model + h5_lowdim_input),
    ),
]

env = dict(
    type="Ros2Env",
    num_queries=chunk_size,
    norm_stats_cache=norm_stats_cache,
    load_pipeline=test_load_pipeline,
    action_dim=58,
    robot_type="v3",
    inference_type="v2",
    obs_timestamp_key=None,
    inference_dt=0.1,
    action_output=h5_action_input,
    temporal_agg=[
        dict(action_name="/action/right_arm/joint_angle",  agg_flag=True, train_dt=0.01666, offset=0, output_chunk=40, with_raw=True),
        dict(action_name="/action/right_hand/joint_angle", agg_flag=True, train_dt=0.01666, offset=0, output_chunk=40),
        dict(action_name="/action/left_arm/joint_angle",   agg_flag=True, train_dt=0.01666, offset=0, output_chunk=40, with_raw=True),
        dict(action_name="/action/left_hand/joint_angle",  agg_flag=True, train_dt=0.01666, offset=0, output_chunk=40),
        dict(action_name="/action/neck/joint_angle",       agg_flag=True, train_dt=0.01666, offset=0, output_chunk=40),
    ],
    debug=False,
)

cfg = dict(env=env)