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
from scipy.ndimage import gaussian_filter
import os
import time
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
import argparse
from stable_baselines3 import PPO
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

def load_ppo_from_hub(env_id="PongNoFrameskip-v4", env=None):
    """Load the PPO model directly from Hugging Face."""
    # Define the model path on Hugging Face
    repo_id = f"sb3/ppo-{env_id}"
    filename = f"ppo-{env_id}.zip"  # Standard filename for PPO models on Hugging Face
    
    print(f"Loading model from Hugging Face: {repo_id}")
    
    try:
        # Download and load the checkpoint
        checkpoint_path = load_from_hub(
            repo_id=repo_id,
            filename=filename
        )
        
        # Load the model
        model = PPO.load(
        checkpoint_path,
        env=env,
        custom_objects={
        # all ReplayBuffer arguments end up here
        "replay_buffer_kwargs": {"handle_timeout_termination": False}
        },
        device="cuda:2"          # or "cuda:1"
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

def run_unlearning(
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
    trajectories_len = [100, 250, 500, 750, 990] if trajectories_len is None else trajectories_len
    modes            = ["uniform"] if modes is None else modes
    repetitions      = max(1, repetitions)  

    

    env = make_atari_env(env_id)
    
    all_results: list[dict[str, Any]] = []
    seed = -1
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

                    agent = load_ppo_from_hub(env_id=env_id, env=env)
                    
                    forget_states = []

                    while len(forget_states) != states_n:   
                        forget_states = []
                        forget_actions = []
                        seed += 1
                        env.venv.seed(seed)
                        state= env.reset()
        
                        for step in range(2000):
                            # Store state
                            forget_states.append(state.copy())
                            #env_state = env.unwrapped.sim.get_state()
                            #env_states.append(env_state)
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
                    

    
                    estimator = RetainSetGenerator(
                        agent=agent, 
                        environment=env,
                        forget_states=forget_states,
                        similarity_threshold=combo["retain"],
                        embedding_dim=512
                    )
                    retain_states, retain_actions = estimator.estimate_retain_set_via_environment(n_trajectories=combo["retain_dim"], max_steps=combo['steps'], random_action_prob=0.001)
                    retain_states = retain_states.squeeze(1)
                    retain_actions = retain_actions.squeeze(1)

                    start = time.perf_counter()
            
                    new_agent, metrics1 = unlearn_policy_kl(
                        agent=agent,
                        forget_states=forget_states,
                        forget_actions=forget_actions,
                        retain_states=retain_states,
                        retain_actions=retain_actions,
                        method=mode,
                        temperature=combo["temperature"],      # High temperature for stability
                        lambda1=combo["lambda1"],            # Lower weight for stability
                        lambda2=combo["lambda2"],           # Lower weight for stability
                        num_steps=1000,          # Fewer steps with full batch
                        lr=combo["learning_rate"],              # Low learning rate
                        min_lr=combo["learning_rate"],
                        max_grad_norm=5.0,      # Gradient clipping
                        clip_q_values=True,     # Scale Q-values
                        device="cuda:2"
                    )

                    elapsed_time = time.perf_counter() - start
                    torch.cuda.empty_cache()
                    if env_id == "SpaceInvadersNoFrameskip-v4" or env_id == "BreakoutNoFrameskip-v4":
                        metrics2 = evaluate(agent, env, n_episodes=20, steps = 4000)
                    else:
                        metrics2 = evaluate(agent, env, n_episodes=20)

                    all_results.append(
                        {
                            "mode": mode,
                            "traj_len": states_n,
                            "seed": seed,
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
    """
    run_unlearning(env_id="PongNoFrameskip-v4", param_grid={
                "lambda1": [1],
                "lambda2": [1],
                "retain": [0.001],
                "retain_dim": [3],
                "temperature": [1],
                "scale_factor": [0.1],
                "learning_rate": [0.0005],
                "steps": [1000]},output_path="./pong-final-seed_results.json",repetitions=10)
    """
    run_unlearning(env_id="SpaceInvadersNoFrameskip-v4", param_grid={
                "lambda1": [1],
                "lambda2": [10],
                "retain": [0.001],
                "retain_dim": [70],
                "temperature": [1],
                "scale_factor": [10],
                "learning_rate": [0.0005],
                "steps": [2000]},output_path="./spaceinvaders-final-seed-uniform_results.json", repetitions=10)
    """
    run_unlearning(env_id="EnduroNoFrameskip-v4", param_grid={
                "lambda1": [1],
                "lambda2": [10],
                "retain": [0.001],
                "retain_dim": [10],
                "temperature": [1],
                "scale_factor": [10],
                "learning_rate": [0.0005],
                "steps": [10_000]} ,output_path="./enduro-final-seed_results.json", repetitions=10)
    
    run_unlearning(env_id="BreakoutNoFrameskip-v4", param_grid={
                "lambda1": [1],
                "lambda2": [10],
                "retain": [0.001],
                "retain_dim": [60],
                "temperature": [1],
                "scale_factor": [10],
                "learning_rate": [0.0005],
                "steps": [1_000]} ,output_path="./breakout-final-seed_results.json", repetitions=10)
                """