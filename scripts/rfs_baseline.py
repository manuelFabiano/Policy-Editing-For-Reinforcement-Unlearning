import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Dict, Optional, Tuple, List
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
import time
import csv
import joblib
from jtop import jtop
import statistics
from threading import Thread

class UnlearningPenaltyWrapper(gym.Wrapper):
    """
    Environment wrapper that penalizes the agent when it chooses forget actions
    in forget states
    """
    def __init__(self, env, forget_states: np.ndarray, forget_actions: np.ndarray,
                 penalty: float = -10.0, state_threshold: float = 0.1):
        super().__init__(env)
        self.forget_states = forget_states
        self.forget_actions = forget_actions
        self.penalty = penalty
        self.state_threshold = state_threshold
        
        # For tracking
        self.penalties_applied = 0
        self.total_steps = 0
        
    def step(self, action):
        # Normal step
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        
        # Check if we're close to a forget state
        if self._is_forget_state(self.last_obs, action):
            reward += self.penalty  # Apply penalty
            self.penalties_applied += 1
            info['unlearning_penalty'] = True
        else:
            info['unlearning_penalty'] = False
            
        # Save for next step
        self.last_obs = obs
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_obs = obs
        return obs, info
    
    def _is_forget_state(self, state, action):
        """Check if state-action is in the forget list"""
        # For continuous actions
        if isinstance(self.action_space, gym.spaces.Box):
            for f_state, f_action in zip(self.forget_states, self.forget_actions):
                state_dist = np.linalg.norm(state - f_state)
                if state_dist < self.state_threshold:
                    action_dist = np.linalg.norm(action - f_action)
                    if action_dist < 0.3:  # Threshold for actions
                        return True
        # For discrete actions
        else:
            for f_state, f_action in zip(self.forget_states, self.forget_actions):
                state_dist = np.linalg.norm(state - f_state)
                if state_dist < self.state_threshold and action == f_action:
                    return True
        return False
    
    def get_penalty_rate(self):
        """Return the rate of applied penalties"""
        if self.total_steps == 0:
            return 0.0
        return self.penalties_applied / self.total_steps
    

class UnlearningCallback(BaseCallback):
    """
    Callback to monitor unlearning during training
    """
    def __init__(self, forget_states, forget_actions, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.forget_states = forget_states
        self.forget_actions = forget_actions
        self.eval_freq = eval_freq
        self.evaluations_forget_accuracy = []
        self.evaluations_timesteps = []
        
    def _on_step(self) -> bool:
        # Evaluate every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            forget_accuracy = self._evaluate_forget_accuracy()
            self.evaluations_forget_accuracy.append(forget_accuracy)
            self.evaluations_timesteps.append(self.num_timesteps)
            
            if self.verbose > 0:
                print(f"Timesteps: {self.num_timesteps}, "
                      f"Forget Accuracy: {forget_accuracy:.2%}")
                
                # Check penalty rate if wrapper is present
                if hasattr(self.training_env.envs[0], 'get_penalty_rate'):
                    penalty_rate = self.training_env.envs[0].get_penalty_rate()
                    print(f"  Penalty Rate: {penalty_rate:.2%}")
        
        return True
    
    def _evaluate_forget_accuracy(self):
        """Evaluate how well the agent avoids forget actions"""
        model = self.model
        correct_avoidance = 0
        
        for state, forget_action in zip(self.forget_states, self.forget_actions):
            # Predict action
            action, _ = model.predict(state, deterministic=True)
            
            # Check if it avoids forget action
            if isinstance(model.action_space, gym.spaces.Box):
                # Continuous: distance from action
                distance = np.linalg.norm(action - forget_action)
                if distance > 0.5:  # Threshold
                    correct_avoidance += 1
            else:
                # Discrete: different action
                if action != forget_action:
                    correct_avoidance += 1
        
        return correct_avoidance / len(self.forget_states)


def train_from_scratch_with_unlearning(
    env_name: str,
    algorithm: str,  # "PPO", "SAC", "TD3"
    forget_states: np.ndarray,
    forget_actions: np.ndarray,
    retain_states: np.ndarray,
    retain_actions: np.ndarray,
    total_timesteps: int = 1000000,
    penalty: float = -10.0,
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    device: str = "auto",
    seed: int = 0,
    verbose: int = 1
) -> Tuple[any, Dict]:
    """
    Train from scratch with penalty for unlearning
    
    Args:
        env_name: Environment name
        algorithm: Algorithm to use
        forget_states: States to forget
        forget_actions: Actions to forget
        retain_states: States to retain
        retain_actions: Actions to retain
        total_timesteps: Total training steps
        penalty: Penalty for forget state-actions
        eval_freq: Evaluation frequency
        n_eval_episodes: Episodes for evaluation
        device: Device for training
        seed: Random seed
        verbose: Verbosity level
        
    Returns:
        model: Trained model
        results: Dictionary with metrics
    """
    
    # Create environment with wrapper
    def make_env():
        env = gym.make(env_name)
        env = UnlearningPenaltyWrapper(env, forget_states, forget_actions, penalty)
        return env
    
    # Create vec env
    env = make_vec_env(make_env, n_envs=1, seed=seed)
    
    # Create separate eval environment (without penalty)
    eval_env = gym.make(env_name)
    
    # Select algorithm
    if algorithm == "PPO":
        model = PPO("MlpPolicy", env, 
                   learning_rate=3e-4,
                   n_steps=2048,
                   batch_size=64,
                   n_epochs=10,
                   gamma=0.99,
                   gae_lambda=0.95,
                   clip_range=0.2,
                   ent_coef=0.01,
                   tensorboard_log=f"./logs/{algorithm}_unlearn",
                   device=device,
                   verbose=verbose)
                   
    elif algorithm == "SAC":
        model = SAC("MlpPolicy", env,
                   learning_rate=3e-4,
                   buffer_size=1000000,
                   learning_starts=10000,
                   batch_size=100,
                   tau=0.005,
                   gamma=0.99,
                   train_freq=1,
                   tensorboard_log=f"./logs/{algorithm}_unlearn",
                   device=device,
                   verbose=verbose)
                   
    elif algorithm == "TD3":
        model = TD3("MlpPolicy", env,
                   learning_rate=0.0003,
                   learning_starts=10000,
                   batch_size=100,
                   tau=0.005,
                   gamma=0.99,
                   train_freq=1,
                   policy_delay=2,
                   tensorboard_log=f"./logs/{algorithm}_unlearn",
                   device=device,
                   verbose=verbose)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Callbacks
    unlearn_callback = UnlearningCallback(forget_states, forget_actions, 
                                        eval_freq=eval_freq, verbose=verbose)
    
    eval_callback = EvalCallback(eval_env, 
                               best_model_save_path=f"./models/{algorithm}_unlearn",
                               log_path=f"./logs/{algorithm}_unlearn",
                               eval_freq=eval_freq,
                               n_eval_episodes=n_eval_episodes,
                               deterministic=True)
    
    # Training
    print(f"\nStarting training from scratch with unlearning penalty...")
    print(f"Algorithm: {algorithm}")
    print(f"Environment: {env_name}")
    print(f"Forget states: {len(forget_states)}")
    print(f"Penalty: {penalty}")
    
    model.learn(total_timesteps=total_timesteps, 
                callback=[unlearn_callback, eval_callback],
                progress_bar=True)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    
    # 1. General performance
    mean_reward, std_reward = evaluate_policy(model, eval_env, 
                                            n_eval_episodes=30,
                                            deterministic=True)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # 2. Forget accuracy
    forget_accuracy = evaluate_forget_accuracy(model, forget_states, 
                                             forget_actions, env.envs[0].action_space)
    print(f"Forget accuracy: {forget_accuracy:.2%}")
    
    # 3. Retain accuracy (if available)
    if retain_states is not None and len(retain_states) > 0:
        retain_accuracy = evaluate_retain_accuracy(model, retain_states, 
                                                 retain_actions, env.envs[0].action_space)
        print(f"Retain accuracy: {retain_accuracy:.2%}")
    else:
        retain_accuracy = None
    
    # Results
    results = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'forget_accuracy': forget_accuracy,
        'retain_accuracy': retain_accuracy,
        'forget_accuracies': unlearn_callback.evaluations_forget_accuracy,
        'timesteps': unlearn_callback.evaluations_timesteps,
        'penalty_rate': env.envs[0].get_penalty_rate() if hasattr(env.envs[0], 'get_penalty_rate') else None
    }
    
    # Plot results
    plot_training_results(results, algorithm)
    
    env.close()
    eval_env.close()
    
    return model, results


def evaluate_forget_accuracy(model, forget_states, forget_actions, action_space):
    """Evaluate accuracy in forgetting"""
    correct = 0
    
    for state, forget_action in zip(forget_states, forget_actions):
        action, _ = model.predict(state, deterministic=True)
        
        if isinstance(action_space, gym.spaces.Box):
            distance = np.linalg.norm(action - forget_action)
            if distance > 0.5:
                correct += 1
        else:
            if action != forget_action:
                correct += 1
                
    return correct / len(forget_states)


def evaluate_retain_accuracy(model, retain_states, retain_actions, action_space):
    """Evaluate accuracy in retaining behaviors"""
    correct = 0
    
    for state, retain_action in zip(retain_states, retain_actions):
        action, _ = model.predict(state, deterministic=True)
        
        if isinstance(action_space, gym.spaces.Box):
            distance = np.linalg.norm(action - retain_action)
            if distance < 0.3:  # Stricter threshold for retain
                correct += 1
        else:
            if action == retain_action:
                correct += 1
                
    return correct / len(retain_states)


def plot_training_results(results, algorithm):
    """Visualize training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Forget accuracy over time
    timesteps = results['timesteps']
    forget_accs = results['forget_accuracies']
    
    ax1.plot(timesteps, forget_accs, linewidth=2, label='Forget Accuracy')
    ax1.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Target 90%')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Forget Accuracy')
    ax1.set_title(f'{algorithm} - Unlearning Progress')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Summary stats
    final_stats = {
        'Mean Reward': results['mean_reward'],
        'Forget Acc': results['forget_accuracy'],
        'Retain Acc': results['retain_accuracy'] if results['retain_accuracy'] else 0,
        'Penalty Rate': results['penalty_rate'] if results['penalty_rate'] else 0
    }
    
    keys = list(final_stats.keys())
    values = list(final_stats.values())
    
    bars = ax2.bar(keys, values)
    ax2.set_ylabel('Value')
    ax2.set_title('Final Metrics')
    ax2.set_ylim(0, max(max(values) * 1.1, 1.0))
    
    # Color bars
    colors = ['blue', 'green', 'orange', 'red']
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add values above bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def load_td3_from_hub(env_id="Walker2d-v3", env=None):
    """Load the TD3 model directly from Hugging Face."""
    # Define the model path on Hugging Face
    repo_id = f"sb3/td3-{env_id}"
    filename = f"td3-{env_id}.zip"  # Standard filename for TD3 models on Hugging Face
    
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
        device="cuda"         
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

from huggingface_sb3 import load_from_hub

env = gym.make("HalfCheetah-v5")
agent = load_td3_from_hub(env_id="HalfCheetah-v3", env=env)

forget_states = []
forget_actions = []
env_states = []
episode_reward = 0
steps = 0 
state, _ = env.reset()
           
episode_reward = 0

for step in range(1500):
    # Store state
    forget_states.append(state.copy())
    #env_state = env.unwrapped.sim.get_state()
    #env_states.append(env_state)
    # Policy action
    action, _ = agent.predict(state, deterministic=True)
                
    # Store action
    forget_actions.append(action.copy() if isinstance(action, np.ndarray) else action)
                
    # Step environment
    next_state, reward, done, truncated, _ = env.step(action)
    done = done or truncated
    episode_reward += reward
    state = next_state
    steps += 1
                
    if done:
        break
        
forget_states = np.array(forget_states)
forget_actions = np.array(forget_actions)
#env_states = np.array(env_states)

forget_states = forget_states[50:]
forget_actions = forget_actions[50:]
print(episode_reward)
print(steps)

import time
import joblib
from jtop import jtop
import statistics

# 1. Initialize energy log
energy_log = []
start_time = time.time()

# 2. Function to monitor energy
def monitor_energy(jetson, keep_running_flag):
    while keep_running_flag["run"]:
        data = jetson.stats
        timestamp = time.time() - start_time
        row = {
                "timestamp": timestamp,
                "VDD_CPU_GPU_CV": data.get("Power VDD_CPU_GPU_CV", 0),
                "VDD_SOC": data.get("Power VDD_SOC", 0),
                "Power_TOTAL": data.get("Power TOT", 0)
            }
        energy_log.append(row)
        time.sleep(1)

# 3. Start jtop and energy monitoring
with jtop() as jetson:
    if jetson.ok():
        keep_running = {"run": True}
        energy_thread = Thread(target=monitor_energy, args=(jetson, keep_running))
        energy_thread.start()

        # 4. Execute training
        model, results = train_from_scratch_with_unlearning(
            env_name="HalfCheetah-v5",
            algorithm="TD3",
            forget_states=forget_states,
            forget_actions=forget_actions,
            retain_states=None,
            retain_actions=None,
            total_timesteps=1000000.0,
            penalty=-10.0,
            eval_freq=20000
        )

        # 5. Stop monitoring
        keep_running["run"] = False
        energy_thread.join()

end_time = time.time()

# 6. Calculate average power
avg_power = {}
for key in ["VDD_CPU_GPU_CV", "VDD_SOC", "Power_TOTAL"]:
    values = [entry[key] for entry in energy_log if entry[key] is not None]
    avg_power[key] = round(statistics.mean(values), 2) if values else 0

# 7. Save model with Stable-Baselines3
model.save("trained_model_sb3")

# 8. Export energy CSV
with open("energy_log.csv", "w", newline="") as csvfile:
    fieldnames = ["timestamp", "VDD_CPU_GPU_CV", "VDD_SOC", "Power_TOTAL"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(energy_log)

# 9. Print summary
print(f"Total time: {round(end_time - start_time, 2)} seconds")
print("Average energy consumption (mW):")
for k, v in avg_power.items():
    print(f"  {k}: {v} mW")
print("📁 Data saved: trained_model_sb3.zip, energy_log.csv")