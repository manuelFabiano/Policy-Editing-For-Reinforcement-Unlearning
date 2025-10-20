import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
from stable_baselines3 import TD3, PPO, SAC
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from gymnasium.wrappers import TransformObservation
from huggingface_sb3 import load_from_hub
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from sklearn.metrics.pairwise import euclidean_distances
from typing import Optional, Tuple, List, Any
import os
import matplotlib.pyplot as plt
import torch
import time
import itertools
import json
from pathlib import Path 
os.environ['LD_LIBRARY_PATH'] = (
    os.environ.get('LD_LIBRARY_PATH', '') + 
    ':/home/user/.mujoco/mujoco210/bin:/usr/lib/nvidia'
)

os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
from unlearning import evaluate_unlearning, unlearn_continuous_policy_kl, evaluate
from RetainSetGenerator import RetainSetGenerator

def load_td3_from_hub(env_id="Walker2d-v3", env=None):
    """Load the TD3 model directly from Hugging Face."""
    # Define the model path on Hugging Face
    repo_id = f"sb3/td3-{env_id}"
    filename = f"td3-{env_id}.zip"  # Standard filename for PPO models on Hugging Face
    
    print(f"Loading model from Hugging Face: {repo_id}")
    
    try:
        # Download and load the checkpoint
        checkpoint_path = load_from_hub(
            repo_id=repo_id,
            filename=filename
        )
        
        # Load the model
        model = TD3.load(
        checkpoint_path,
        env=env,
        custom_objects={
        # all ReplayBuffer arguments end up here
        "replay_buffer_kwargs": {"handle_timeout_termination": False}
        },
        device="cuda:2"         
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
    except ImportError:
        pass  # numpy not available / not needed
    # Let the default encoder handle (will raise TypeError if still unserialisable)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")



def run_unlearning(
    *,
    env_id: str = "Walker2d-v3",
    param_grid: dict[str, list[Any]] | None = None,
    trajectories_len: list[int] | None = None,
    modes: list[str] | None = None,
    repetitions: int = 1,
    output_path: str | Path = "./unlearning_results.json",
):
    """
    if param_grid is None:
        param_grid = {
            "lambda1":      [1],
            "lambda2":      [5, 10, 20, 40],
            "retain":       [0.1, 0.25, 0.5,1.0, 1.5],
            "retain_dim":   [100, 150, 300],
            "retain_rand":  [0.05, 0.1],
            "temperature":  [1, 2.5, 5]
        }
    """
   
    
    random.seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)

    trajectories_len = [100, 250, 500, 750, 990] if trajectories_len is None else trajectories_len
    modes            = ["random"] if modes is None else modes
    repetitions      = max(1, repetitions)  

    seed = -1
    env = gym.make(env_id, render_mode="rgb_array")
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

                    agent = load_td3_from_hub(env_id=env_id, env=env)
                    
                    forget_states = []
                    
                    # Sample forget trajectory:
                    while len(forget_states) != states_n:   
                        forget_states = []
                        forget_actions = []
                        seed += 1
                        state, _ = env.reset(seed=seed)  
                        for step in range(1500):
                            # Store state
                            forget_states.append(state.copy())
                            
                            # Policy action
                            action, _ = agent.predict(state, deterministic=True)
                                        
                            # Store action
                            forget_actions.append(action.copy() if isinstance(action, np.ndarray) else action)
                                        
                            # Step environment
                            next_state, reward, done, truncated, _ = env.step(action)
                            done = done or truncated
                            state = next_state
                                        
                            if done:
                                break
                                
                        forget_states = np.array(forget_states)
                        forget_actions = np.array(forget_actions)
                        if len(forget_states) < states_n + 10:
                            continue
                        s = random.randint(10, len(forget_states)- states_n)
                        forget_states = forget_states[s:s+ states_n]
                        forget_actions = forget_actions[s:s+ states_n]



                    estimator = RetainSetGenerator(env, agent, forget_states)
                    retain_states, retain_actions = estimator.distance_aware_sampling(
                        combo["retain_dim"],
                        2000,
                        min_distance=combo["retain"],
                        random_action_prob=combo["retain_rand"],
                    )

                    start = time.perf_counter()
                    new_agent, metrics1, _, _ = unlearn_continuous_policy_kl(
                        agent,
                        forget_states,
                        forget_actions,
                        retain_states,
                        retain_actions,
                        method=mode,          
                        temperature1=combo["temperature1"],
                        temperature2=combo["temperature2"],
                        lambda1=combo["lambda1"],
                        lambda2=combo["lambda2"],
                        num_steps=10_000,
                        lr=0.0005,
                        min_lr=0.0005,
                        max_grad_norm=1,
                        device="cuda:2"
                    )
                    elapsed_time = time.perf_counter() - start
                    metrics2 = evaluate(agent, env, n_episodes=100)

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
    
    run_unlearning(env_id="HalfCheetah-v3", param_grid={
            "lambda1": [1],
            "lambda2": [3],
            "retain": [1.5],
            "retain_dim": [300],
            "retain_rand": [0.1],
            "temperature1": [1],
            "temperature2": [1]
        }, output_path="./halfcheetah-final-seed-random_results.json", repetitions=10)
    
    run_unlearning(env_id="Walker2d-v3", param_grid={
            "lambda1": [1],
            "lambda2": [5],
            "retain": [1.5],
            "retain_dim": [150],
            "retain_rand": [0.1],
            "temperature1": [3],
            "temperature2": [3]}, output_path="./walker2d-final-seed-random_results.json", repetitions=10)
    
    run_unlearning(env_id="Hopper-v3", param_grid={
            "lambda1": [1],
            "lambda2": [2],
            "retain": [0.5],
            "retain_dim": [200],
            "retain_rand": [0.05],
            "temperature1": [2],
            "temperature2": [2]}, output_path="./hopper-final-seed-random_results.json", repetitions=10)