import gymnasium as gym
import ale_py
import pickle
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from sklearn.decomposition import PCA
import cv2
from scipy.ndimage import gaussian_filter, shift
import os
import time
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from huggingface_sb3 import load_from_hub
from stable_baselines3.common.env_util import make_atari_env

from unlearning import evaluate_unlearning, unlearn_policy_kl, evaluate
from RetainSetGenerator import RetainSetGenerator

def make_atari_env(env_id="PongNoFrameskip-v4"):
    """Create a preprocessed Atari environment."""
    def make_env():
        env = gym.make(env_id, render_mode='rgb_array')
        env = AtariWrapper(env, clip_reward=False, terminal_on_life_loss=False)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)
    return env

def load_dqn_from_hub(env_id="PongNoFrameskip-v4", env=None):
    """Load the DQN model directly from Hugging Face."""
    # Define the model path on Hugging Face
    repo_id = f"sb3/dqn-{env_id}"
    filename = f"dqn-{env_id}.zip"  # Standard filename for DQN models on Hugging Face
    
    print(f"Loading model from Hugging Face: {repo_id}")
    
    try:
        # Download and load the checkpoint
        checkpoint_path = load_from_hub(
            repo_id=repo_id,
            filename=filename
        )
        
        # Load the model
        model = DQN.load(
        checkpoint_path,
        env=env,
        custom_objects={
        # all ReplayBuffer arguments end up here
        "replay_buffer_kwargs": {"handle_timeout_termination": False}
        },
        device="cuda:1"          # or "cuda:2"
        )
        print(f"Model loaded successfully from {checkpoint_path}")
        policy = model.policy
        print(f"Policy type: {type(policy)}")
        print(f"Architecture: {policy.features_extractor}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Verify that the model exists on Hugging Face: {repo_id}")
        raise

  
def to_serialisable(obj):
    """Helper to convert non‑JSON‑serialisable objects (e.g. NumPy arrays) to Python lists."""
    try:
        import numpy as np

        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.generic, np.number)):
            return obj.item()  # or float(obj)
    except ImportError:
        pass  # numpy not available / not needed
    # Let the default encoder handle (will raise TypeError if still unserialisable)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def perturb_pong(matrix, max_shift_paddle=10, max_shift_ball=10):
    """
    Intelligently perturb a Pong image.
    
    Parameters:
    - matrix: 84x84 image
    - max_shift_paddle: maximum vertical displacement for paddles
    - max_shift_ball: maximum displacement for ball (both x and y)
    
    Returns:
    - perturbed_matrix: perturbed image
    """
    # Create a copy to not modify the original
    perturbed_matrix = matrix.copy()
    
    # Define areas
    ROW_START = 14
    ROW_END = 77
    
    PADDLE1_COL_START = 0
    PADDLE1_COL_END = 11
    
    BALL_COL_START = 12
    BALL_COL_END = 72
    
    PADDLE2_COL_START = 73
    PADDLE2_COL_END = 84
    
    # 1. Perturb the first paddle (only vertically)
    shift_y_paddle1 = np.random.randint(-max_shift_paddle, max_shift_paddle + 1)
    paddle1_area = matrix[ROW_START:ROW_END, PADDLE1_COL_START:PADDLE1_COL_END]
    paddle1_shifted = shift(paddle1_area, [shift_y_paddle1, 0], mode='nearest')
    perturbed_matrix[ROW_START:ROW_END, PADDLE1_COL_START:PADDLE1_COL_END] = paddle1_shifted
    
    # 2. Perturb the ball area (horizontally and vertically)
    shift_x_ball = np.random.randint(-max_shift_ball, max_shift_ball + 1)
    shift_y_ball = np.random.randint(-max_shift_ball, max_shift_ball + 1)
    ball_area = matrix[ROW_START:ROW_END, BALL_COL_START:BALL_COL_END]
    ball_shifted = shift(ball_area, [shift_y_ball, shift_x_ball], mode='nearest')
    perturbed_matrix[ROW_START:ROW_END, BALL_COL_START:BALL_COL_END] = ball_shifted
    
    # 3. Perturb the second paddle (only vertically)
    shift_y_paddle2 = np.random.randint(-max_shift_paddle, max_shift_paddle + 1)
    paddle2_area = matrix[ROW_START:ROW_END, PADDLE2_COL_START:PADDLE2_COL_END]
    paddle2_shifted = shift(paddle2_area, [shift_y_paddle2, 0], mode='nearest')
    perturbed_matrix[ROW_START:ROW_END, PADDLE2_COL_START:PADDLE2_COL_END] = paddle2_shifted
    
    return perturbed_matrix, {
        'paddle1_shift': shift_y_paddle1,
        'ball_shift': (shift_x_ball, shift_y_ball),
        'paddle2_shift': shift_y_paddle2
    }


def perturb_stacked_state(stacked_state, max_shift_paddle=10, max_shift_ball=10):
    """
    Perturb a stacked Pong state (4 frames) applying the SAME perturbation to all frames.
    
    Parameters:
    - stacked_state: array of shape (4, 84, 84)
    - max_shift_paddle: maximum vertical displacement for paddles
    - max_shift_ball: maximum displacement for ball
    
    Returns:
    - perturbed_state: array of shape (4, 84, 84)
    - perturbation_info: dictionary with info about applied perturbation
    """
    perturbed_state = np.zeros_like(stacked_state)
    
    # Generate a perturbation using the first frame as reference
    _, perturbation_info = perturb_pong(stacked_state[0], max_shift_paddle, max_shift_ball)
    
    # Apply the SAME perturbation to all 4 frames
    for i in range(4):
        # Recreate the perturbation with the same parameters
        perturbed_matrix = stacked_state[i].copy()
        
        # Define areas (as in perturb_pong)
        ROW_START = 14
        ROW_END = 77
        PADDLE1_COL_START = 0
        PADDLE1_COL_END = 11
        BALL_COL_START = 12
        BALL_COL_END = 72
        PADDLE2_COL_START = 73
        PADDLE2_COL_END = 84
        
        # Apply saved shifts
        # Paddle 1
        paddle1_area = stacked_state[i][ROW_START:ROW_END, PADDLE1_COL_START:PADDLE1_COL_END]
        paddle1_shifted = shift(paddle1_area, [perturbation_info['paddle1_shift'], 0], 
                               mode='nearest')
        perturbed_matrix[ROW_START:ROW_END, PADDLE1_COL_START:PADDLE1_COL_END] = paddle1_shifted
        
        # Ball area
        ball_area = stacked_state[i][ROW_START:ROW_END, BALL_COL_START:BALL_COL_END]
        ball_shifted = shift(ball_area, [perturbation_info['ball_shift'][1], 
                                        perturbation_info['ball_shift'][0]], 
                           mode='nearest')
        perturbed_matrix[ROW_START:ROW_END, BALL_COL_START:BALL_COL_END] = ball_shifted
        
        # Paddle 2
        paddle2_area = stacked_state[i][ROW_START:ROW_END, PADDLE2_COL_START:PADDLE2_COL_END]
        paddle2_shifted = shift(paddle2_area, [perturbation_info['paddle2_shift'], 0], 
                               mode='nearest')
        perturbed_matrix[ROW_START:ROW_END, PADDLE2_COL_START:PADDLE2_COL_END] = paddle2_shifted
        
        perturbed_state[i] = perturbed_matrix
    
    return perturbed_state, perturbation_info


def generate_pong_retain_set(forget_states, perturbations_per_state=5,
                           max_shift_paddle_range=(5, 15),
                           max_shift_ball_range=(5, 15),
                           mix_original_states=True,
                           original_state_ratio=0.2,
                           visualize_samples=True):
    """
    Generate a retain set by perturbing forget states.
    
    Parameters:
    - forget_states: array of shape (N, 4, 84, 84) with states to forget
    - perturbations_per_state: number of perturbations per state
    - max_shift_paddle_range: range (min, max) for paddle shift
    - max_shift_ball_range: range (min, max) for ball shift
    - mix_original_states: if True, also include original unperturbed states
    - original_state_ratio: percentage of original states to include
    - visualize_samples: if True, visualize some examples
    
    Returns:
    - retain_states: array of shape (M, 4, 84, 84) with perturbed states
    """
    n_forget = len(forget_states)
    retain_states = []
    perturbation_infos = []
    
    print(f"Generating retain set from {n_forget} forget states...")
    print(f"Perturbations per state: {perturbations_per_state}")
    
    # Generate perturbations for each state
    for idx in tqdm(range(n_forget), desc="Perturbating states"):
        state = forget_states[idx]
        
        for p in range(perturbations_per_state):
            # Vary perturbation parameters for diversity
            max_shift_paddle = np.random.randint(max_shift_paddle_range[0], 
                                               max_shift_paddle_range[1] + 1)
            max_shift_ball = np.random.randint(max_shift_ball_range[0], 
                                             max_shift_ball_range[1] + 1)
            
            # Perturb the stacked state
            perturbed_state, info = perturb_stacked_state(
                state, max_shift_paddle, max_shift_ball
            )
            
            retain_states.append(perturbed_state)
            perturbation_infos.append(info)
    
    # Add original states if requested
    if mix_original_states and original_state_ratio > 0:
        n_original = int(len(retain_states) * original_state_ratio)
        # Select random states from forget set
        original_indices = np.random.choice(n_forget, n_original, replace=True)
        original_states = forget_states[original_indices]
        
        # Add light minimal perturbations for variation
        for state in original_states:
            # Very small perturbation to keep almost original
            perturbed_state, _ = perturb_stacked_state(state, 
                                                     max_shift_paddle=2, 
                                                     max_shift_ball=2)
            retain_states.append(perturbed_state)
        
        print(f"Added {n_original} lightly perturbed original states")
    
    retain_states = np.array(retain_states)
    print(f"Generated {len(retain_states)} retain states")
    
    # Visualize some examples
    if visualize_samples and len(retain_states) > 0:
        n_examples = min(4, len(forget_states))
        fig, axes = plt.subplots(n_examples, 5, figsize=(15, 3*n_examples))
        
        for i in range(n_examples):
            # Show original state (first frame)
            axes[i, 0].imshow(forget_states[i, 0], cmap='gray')
            axes[i, 0].set_title(f'Original {i+1}')
            axes[i, 0].axis('off')
            
            # Show 4 different perturbations
            for j in range(4):
                idx = i * perturbations_per_state + j
                if idx < len(retain_states):
                    axes[i, j+1].imshow(retain_states[idx, 0], cmap='gray')
                    info = perturbation_infos[idx] if idx < len(perturbation_infos) else {}
                    axes[i, j+1].set_title(f'P{j+1}: B{info.get("ball_shift", "?")}', 
                                          fontsize=8)
                    axes[i, j+1].axis('off')
        
        plt.suptitle('Original vs Perturbed States (showing first frame of stack)')
        plt.tight_layout()
        plt.show()
        
        # Show perturbation statistics
        if perturbation_infos:
            paddle1_shifts = [info['paddle1_shift'] for info in perturbation_infos[:100]]
            paddle2_shifts = [info['paddle2_shift'] for info in perturbation_infos[:100]]
            ball_x_shifts = [info['ball_shift'][0] for info in perturbation_infos[:100]]
            ball_y_shifts = [info['ball_shift'][1] for info in perturbation_infos[:100]]
            
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            
            axes[0, 0].hist(paddle1_shifts, bins=20, alpha=0.7)
            axes[0, 0].set_title('Paddle 1 Vertical Shifts')
            axes[0, 0].set_xlabel('Pixels')
            
            axes[0, 1].hist(paddle2_shifts, bins=20, alpha=0.7)
            axes[0, 1].set_title('Paddle 2 Vertical Shifts')
            axes[0, 1].set_xlabel('Pixels')
            
            axes[1, 0].hist(ball_x_shifts, bins=20, alpha=0.7)
            axes[1, 0].set_title('Ball Horizontal Shifts')
            axes[1, 0].set_xlabel('Pixels')
            
            axes[1, 1].hist(ball_y_shifts, bins=20, alpha=0.7)
            axes[1, 1].set_title('Ball Vertical Shifts')
            axes[1, 1].set_xlabel('Pixels')
            
            plt.suptitle('Perturbation Statistics (first 100 samples)')
            plt.tight_layout()
            plt.show()
    
    return retain_states


# Utility function to get actions from model
def get_actions_for_states(agent, states, batch_size=32, device=None):
    """
    Get actions that the agent would choose for given states.
    
    Parameters:
    - agent: the DQN agent
    - states: array of shape (N, 4, 84, 84)
    - batch_size: batch size for inference
    - device: device to use
    
    Returns:
    - actions: array of shape (N,) with chosen actions
    """
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    q_function = agent.policy.q_net
    q_function.eval()
    
    actions = []
    
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            batch = states[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
            
            q_values = q_function(batch_tensor)
            batch_actions = torch.argmax(q_values, dim=1)
            actions.extend(batch_actions.cpu().numpy())
    
    return np.array(actions)


# Complete usage example
def create_retain_set_for_unlearning(agent, forget_states, forget_actions,
                                   retain_multiplier=5,
                                   perturbation_diversity='high'):
    """
    Create a complete retain set for unlearning.
    
    Parameters:
    - agent: the DQN agent
    - forget_states: states to forget
    - forget_actions: actions to forget
    - retain_multiplier: how many times larger the retain set should be
    - perturbation_diversity: 'low', 'medium', or 'high'
    
    Returns:
    - retain_states: retain set states
    - retain_actions: retain set actions
    """
    # Set parameters based on requested diversity
    if perturbation_diversity == 'low':
        perturbations_per_state = 3
        max_shift_paddle_range = (3, 7)
        max_shift_ball_range = (3, 7)
    elif perturbation_diversity == 'medium':
        perturbations_per_state = 5
        max_shift_paddle_range = (5, 10)
        max_shift_ball_range = (5, 10)
    else:  # high
        perturbations_per_state = retain_multiplier
        max_shift_paddle_range = (5, 20)
        max_shift_ball_range = (3, 15)
    
    # Generate the retain set
    retain_states = generate_pong_retain_set(
        forget_states,
        perturbations_per_state=perturbations_per_state,
        max_shift_paddle_range=max_shift_paddle_range,
        max_shift_ball_range=max_shift_ball_range,
        mix_original_states=False,
        original_state_ratio=0.2,
        visualize_samples=True
    )
    
    # Get actions that the model would choose
    print("Getting actions from model for retain states...")
    retain_actions = get_actions_for_states(agent, retain_states)
    
    # Show statistics
    print(f"\nRetain set statistics:")
    print(f"Total retain states: {len(retain_states)}")
    print(f"Retain/Forget ratio: {len(retain_states)/len(forget_states):.1f}x")
    
    unique_actions, counts = np.unique(retain_actions, return_counts=True)
    print("\nAction distribution in retain set:")
    for action, count in zip(unique_actions, counts):
        print(f"  Action {action}: {count/len(retain_actions)*100:.1f}%")
    
    return retain_states, retain_actions


def run_unlearning_grid_search(
    *,
    env_id: str = 'PongNoFrameskip-v4',
    param_grid: dict[str, list[Any]] | None = None,
    trajectories_len: list[int] | None = None,
    modes: list[str] | None = None,
    repetitions: int = 1,
    output_path: str | Path = "./unlearning_results.json",
):
    if param_grid is None:
        param_grid = {
            "lambda1":      [1],
            "lambda2":      [10,20, 50],
            "retain":       [0.25, 0.5,1.0],
            "retain_dim":   [100, 200, 500],
            "retain_rand":  [0.1],
            "temperature":  [ 1, 2]
        }

    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    trajectories_len = [500] if trajectories_len is None else trajectories_len
    modes            = ["opposite"] if modes is None else modes
    repetitions      = max(1, repetitions)  

    
    seed = -1
    env = make_atari_env(env_id)
    
    all_results: list[dict[str, Any]] = []

    param_names = sorted(param_grid.keys())
    for values in itertools.product(*(param_grid[p] for p in param_names)):
        combo = dict(zip(param_names, values))

        for mode in modes:                       
            for states_n in trajectories_len:    
                for rep in range(1, repetitions + 1):  
                    key = (
                        f"{mode}|traj_len={states_n}|"
                        + "|".join(f"{k}={v}" for k, v in combo.items())
                    )
                    print(f"🏃‍♂️  {key}", flush=True)

                    agent = load_dqn_from_hub(env_id=env_id, env=env)
                    
                    forget_states = []

                    while len(forget_states) != states_n:   
                        forget_states = []
                        forget_actions = []
                        seed += 1
                        env.venv.seed(seed)
                        state= env.reset()

                        
                        for step in range(10000):
                            # Store state
                            forget_states.append(state.copy())
                            
                            # Policy action
                            action= agent.predict(state, deterministic=True)[0]
                                        
                            # Store action
                            forget_actions.append(action.copy() if isinstance(action, np.ndarray) else action)
                                        
                            # Step environment
                            next_state, reward, done, _ = env.step(action)

                        
                            state = next_state
                        
                                        
                            if done:
                                break
                                
                        forget_states = np.array(forget_states)
                        forget_actions = np.array(forget_actions)
                        if len(forget_states) < states_n + 10:
                            continue
                        s = random.randint(10, len(forget_states)- states_n)
                        forget_states = forget_states[s:s+ states_n].squeeze(1)
                        forget_actions = forget_actions[s:s+ states_n].squeeze(1)
                        print(len(forget_states))
                    

    
                    retain_states, retain_actions = create_retain_set_for_unlearning(
                        agent, forget_states, forget_actions,
                        perturbation_diversity='high',  # Larger perturbations
                        retain_multiplier=combo["retain_dim"]  # More retain states
                    )

                    start = time.perf_counter()
            
                    new_agent, metrics1 = unlearn_policy_kl(
                        agent=agent,
                        forget_states=forget_states,
                        forget_actions=forget_actions,
                        retain_states=retain_states,
                        retain_actions=retain_actions,
                        method='proportional',
                        temperature=combo["temperature"],      # High temperature for stability
                        lambda1=combo["lambda1"],            # Lower weight for stability
                        lambda2=combo["lambda2"],           # Lower weight for stability
                        num_steps=1500,          # Fewer steps with full batch
                        lr=combo["learning_rate"],              # Low learning rate
                        min_lr=combo["learning_rate"],
                        max_grad_norm=5.0,      # Gradient clipping
                        clip_q_values=True,     # Scale Q-values
                        q_scale_factor=combo["scale_factor"],   # Scale factor
                        device="cuda:1"
                    )

                    elapsed_time = time.perf_counter() - start
                    torch.cuda.empty_cache()
                    if env_id == "BreakoutNoFrameskip-v4":
                        metrics2 = evaluate(agent, env, n_episodes=20, steps=4000)
                    else:
                        metrics2 = evaluate(agent, env, n_episodes=20)

                    all_results.append(
                        {
                            "mode": mode,
                            "traj_len": states_n,
                            "seed":seed,
                            "hyperparams": combo,
                            "elapsed_time": elapsed_time,
                            "metrics1": metrics1,
                            "metrics2": metrics2,
                        }
                    )

    Path(output_path).write_text(
        json.dumps(all_results, default=to_serialisable, indent=4)
    )
    print(f"\n✅ Search completed. Results in {Path(output_path).resolve()}")


if __name__ == "__main__":  
    
    run_unlearning_grid_search(env_id="PongNoFrameskip-v4", param_grid={
                "lambda1": [1],
                "lambda2": [150],
                "retain": [0.00005],
                "retain_dim": [100],
                "temperature": [0.1],
                "scale_factor": [1],
                "learning_rate": [0.0005],
                "steps": [2000]
                },output_path="./pong-final-seed_results.json",repetitions=10)
    """
    run_unlearning_grid_search(env_id="SpaceInvadersNoFrameskip-v4", param_grid={
                "lambda1": [1],
                "lambda2": [15],
                "retain": [0.00001],
                "retain_dim": [50],
                "temperature": [1],
                "scale_factor": [10],
                "learning_rate": [0.0005],
                "steps": [3000]},output_path="./spaceinvaders-final-seed_results.json", trajectories_len=[100, 250, 500, 750], repetitions=10)
    
    
    run_unlearning_grid_search(env_id="EnduroNoFrameskip-v4", param_grid={
                "lambda1": [1],
                "lambda2": [12],
                "retain": [0.001],
                "retain_dim": [10],
                "temperature": [0.1],
                "scale_factor": [10],
                "learning_rate": [0.0005],
                "steps": [10_000]} ,output_path="./enduro-final-seed_results.json", repetitions=10)
    
    run_unlearning_grid_search(env_id="BreakoutNoFrameskip-v4", param_grid={
                "lambda1": [1],
                "lambda2": [15],
                "retain": [0.001],
                "retain_dim": [60],
                "temperature": [1],
                "scale_factor": [10],
                "learning_rate": [0.0001],
                "steps": [1_500]} ,output_path="./breakout-final-seed_results.json", repetitions=10)
       """