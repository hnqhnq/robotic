"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""

import dataclasses
import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.libero.smoothness_metrics import SmoothnessTracker
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK
from bitvla.constants import (
    BITNET_DEFAULT_IMAGE_TOKEN_IDX,
    BITNET_PROPRIO_PAD_IDX,
    BITNET_IGNORE_INDEX,
    BITNET_ACTION_TOKEN_BEGIN_IDX,
    BITNET_STOP_INDEX,
)


# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 520,  # longest training demo has 505 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps: int = 50                    # (When `diffusion==True`) Number of diffusion steps for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy
    execution_mode: str = "chunk"                    # "chunk": execute first num_open_loop_steps of each chunk;
                                                     # "te": ACT temporal ensembling (query every step);
                                                     # "blend": linear blend of new chunk head with previous chunk's
                                                     #          unexecuted tail over the overlap (ChunkFlow-style)
    te_decay: float = 0.01                           # (te) weights exp(-te_decay * i), i = 0 for the oldest prediction

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL  # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    task_ids: str = ""                               # Comma-separated task ids to evaluate (e.g. "0,1,2"); "" = all tasks.
                                                     #   Used to shard one eval across several GPUs/processes.
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = True                          # Whether to also log results in Weights & Biases
    wandb_entity: str = ""          # Name of WandB entity
    wandb_project: str = ""        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)
    info_in_path: str = ""                           # the info need to inject into the path to identify the result

    save_smoothness_json: bool = True                # Save action smoothness metrics to JSON
    smoothness_json_path: str = ""                   # Optional output path (default: local_log_dir/smoothness_<run_id>.json)
    save_raw_npz: bool = True                        # Save per-episode raw chunks / executed actions / EEF pose (npz)
    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"

    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"

    assert cfg.execution_mode in ("chunk", "te", "blend"), f"Invalid execution_mode: {cfg.execution_mode}"
    assert 1 <= cfg.num_open_loop_steps <= NUM_ACTIONS_CHUNK, "num_open_loop_steps must be in [1, NUM_ACTIONS_CHUNK]"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)

    if cfg.model_family in ["bitnet"]:
        hidden_size  = model.config.text_config.hidden_size
    elif cfg.model_family == "openvla":
        hidden_size = model.llm_dim
    else:
        raise ValueError(f"Unsupported model family: {cfg.model_family}")
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            hidden_size,
            proprio_dim=8,  # 8-dimensional proprio for LIBERO
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression or cfg.use_diffusion:
        action_head = get_action_head(cfg, hidden_size)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None
    if cfg.use_diffusion:
        noisy_action_projector = get_noisy_action_projector(cfg, hidden_size)

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family in ["openvla","bitnet"]:
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging if enabled
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def process_action(action, model_family):
    """Process action before sending to environment."""
    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if model_family in ["openvla","bitnet"]:
        action = invert_gripper_action(action)

    return action


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    initial_state=None,
    log_file=None,
    smoothness_tracker: Optional[SmoothnessTracker] = None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    # The policy is always queried for the full chunk; execution_mode decides what is executed.
    full_chunk_cfg = dataclasses.replace(cfg, num_open_loop_steps=NUM_ACTIONS_CHUNK)
    exec_steps = 1 if cfg.execution_mode == "te" else cfg.num_open_loop_steps
    action_queue = deque()
    prev_chunk = None                  # (blend) last full chunk
    te_buffer = []                     # (te) list of (query_step, full_chunk), oldest first

    # Setup
    t = 0
    replay_images = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]
    raw = {"chunks": [], "query_steps": [], "exec_actions": [], "env_actions": [],
           "eef_pos": [], "eef_quat": [], "gripper_qpos": []}

    # Run episode
    success = False
    if smoothness_tracker is not None:
        smoothness_tracker.start_episode()
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            raw["eef_pos"].append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
            raw["eef_quat"].append(np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
            raw["gripper_qpos"].append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                chunk = np.asarray(
                    get_action(
                        full_chunk_cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    ),
                    dtype=np.float64,
                )
                raw["chunks"].append(chunk)
                raw["query_steps"].append(t - cfg.num_steps_wait)
                if smoothness_tracker is not None:
                    smoothness_tracker.add_chunk(chunk)

                if cfg.execution_mode == "chunk":
                    segment = chunk[:exec_steps]
                elif cfg.execution_mode == "blend":
                    segment = chunk[:exec_steps].copy()
                    if prev_chunk is not None:
                        overlap = min(exec_steps, NUM_ACTIONS_CHUNK - exec_steps)
                        for j in range(overlap):
                            w = 0.0 if overlap == 1 else j / (overlap - 1)
                            segment[j] = w * chunk[j] + (1.0 - w) * prev_chunk[exec_steps + j]
                    prev_chunk = chunk
                else:  # te
                    now = t - cfg.num_steps_wait
                    te_buffer.append((now, chunk))
                    te_buffer = [(q, c) for q, c in te_buffer if now - q < NUM_ACTIONS_CHUNK]
                    preds = np.stack([c[now - q] for q, c in te_buffer])
                    weights = np.exp(-cfg.te_decay * np.arange(len(preds)))
                    segment = (weights[:, None] * preds).sum(axis=0, keepdims=True) / weights.sum()
                action_queue.extend(segment)

            # Get action from queue
            action = action_queue.popleft()
            raw["exec_actions"].append(np.asarray(action, dtype=np.float64))

            # Process action
            action = process_action(action, cfg.model_family)
            raw["env_actions"].append(np.asarray(action, dtype=np.float64))

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    if smoothness_tracker is not None:
        smoothness_tracker.end_episode(success)

    return success, replay_images, raw


def save_raw_episode(path: Path, raw: dict, success: bool, task_id: int, episode_idx: int, cfg: GenerateConfig) -> None:
    """Save one episode's raw predicted chunks, executed actions and EEF trajectory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, v in raw.items():
        arrays[k] = np.stack(v) if len(v) > 0 else np.zeros((0,))
    np.savez_compressed(
        path,
        **arrays,
        success=np.bool_(success),
        task_id=np.int64(task_id),
        episode_idx=np.int64(episode_idx),
        num_open_loop_steps=np.int64(cfg.num_open_loop_steps),
        execution_mode=np.str_(cfg.execution_mode),
        te_decay=np.float64(cfg.te_decay),
    )


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    total_episodes=0,
    total_successes=0,
    log_file=None,
    smoothness_tracker: Optional[SmoothnessTracker] = None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        log_message(f"\nTask: {task_description}", log_file)
        log_message(f"Task id: {task_id}", log_file)

        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            # Use default initial state
            initial_state = initial_states[episode_idx]
        else:
            # Get keys for fetching initial episode state from JSON
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"

            # Skip episode if expert demonstration failed to complete the task
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file)
                continue

            # Get initial state
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])

        log_message(f"Starting episode {task_episodes + 1}...", log_file)

        # Run episode
        success, replay_images, raw = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            smoothness_tracker,
        )

        if cfg.save_raw_npz:
            save_raw_episode(
                Path(cfg.local_log_dir) / "raw" / f"task{task_id}_ep{episode_idx:02d}.npz",
                raw, success, task_id, episode_idx, cfg,
            )

        # Update counters
        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes += 1
            total_successes += 1

        # Save replay video
        save_rollout_video(
            cfg.info_in_path,replay_images, total_episodes, success=success, task_description=task_description, log_file=log_file
        )

        # Log results
        log_message(f"Success: {success}", log_file)
        log_message(f"# episodes completed so far: {total_episodes}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)", log_file)

    # Log task results
    task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    log_message(f"Current task success rate: {task_success_rate}", log_file)
    log_message(f"Current total success rate: {total_success_rate}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                f"success_rate/{task_description}": task_success_rate,
                f"num_episodes/{task_description}": task_episodes,
            }
        )

    return total_episodes, total_successes


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    cfg.local_log_dir = os.path.join(cfg.local_log_dir,cfg.info_in_path)
    
    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize model and components
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    
    if cfg.model_family == "bitnet":
        model.set_constant(
            image_token_idx = BITNET_DEFAULT_IMAGE_TOKEN_IDX,
            proprio_pad_idx = BITNET_PROPRIO_PAD_IDX,
            ignore_idx = BITNET_IGNORE_INDEX,
            action_token_begin_idx = BITNET_ACTION_TOKEN_BEGIN_IDX,
            stop_index = BITNET_STOP_INDEX,
    )
            
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    smoothness_tracker = SmoothnessTracker() if cfg.save_smoothness_json else None

    # Start evaluation
    total_episodes, total_successes = 0, 0
    if cfg.task_ids.strip():
        selected_task_ids = [int(x) for x in cfg.task_ids.split(",") if x.strip() != ""]
        assert all(0 <= t < num_tasks for t in selected_task_ids), f"task_ids out of range [0, {num_tasks})"
    else:
        selected_task_ids = list(range(num_tasks))
    log_message(f"Evaluating task ids: {selected_task_ids}", log_file)

    for task_id in tqdm.tqdm(selected_task_ids):
        total_episodes, total_successes = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            total_episodes,
            total_successes,
            log_file,
            smoothness_tracker,
        )

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if smoothness_tracker is not None and smoothness_tracker.episodes:
        agg = smoothness_tracker.aggregate()
        log_message(f"Smoothness metrics: {json.dumps(agg, indent=2)}", log_file)
        smoothness_path = cfg.smoothness_json_path or os.path.join(
            cfg.local_log_dir, f"smoothness_{run_id}.json"
        )
        smoothness_tracker.save_json(Path(smoothness_path))
        log_message(f"Saved smoothness metrics to {smoothness_path}", log_file)

    # Log to wandb if enabled
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": final_success_rate,
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
