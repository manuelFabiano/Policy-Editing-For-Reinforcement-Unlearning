import numpy as np
from typing import Optional, Dict, Tuple, Union
import matplotlib.pyplot as plt


class RetainSetGenerator:
    """
    Generate retain sets for continuous control unlearning
    """
    
    def __init__(self, env, agent, forget_states = None):
        self.env = env
        self.agent = agent
        self.forget_states = forget_states
    
    def set_forget_states(self, forget_states: np.ndarray):
        """
        Set the states to forget (needed for distance-based sampling)
        
        Args:
            forget_states: Array of states to forget, shape (n_states, state_dim)
        """
        self.forget_states = forget_states
        print(f"Set {len(forget_states)} forget states")
    
    
    def distance_aware_sampling(
        self,
        n_trajectories: int = 30,
        max_steps: int = 200,
        min_distance: float = 0.5,
        distance_metric: str = 'euclidean',
        random_action_prob: float = 0.2,
        deterministic: bool = True,
        normalize_states: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Distance-aware sampling: only collect states that are far enough from forget states
        
        Args:
            n_trajectories: Number of trajectories to attempt
            max_steps: Maximum steps per trajectory
            min_distance: Minimum distance from forget states
            distance_metric: Distance metric to use ('euclidean', 'cosine', 'manhattan')
            random_action_prob: Probability of taking random actions
            deterministic: Whether to use deterministic policy
            normalize_states: Whether to normalize states before computing distances
            
        Returns:
            retain_states: Collected states (far from forget states)
            retain_actions: Collected actions
        """
        if self.forget_states is None:
            raise ValueError("forget_states must be set before distance-aware sampling")
        
        print("\n=== Distance-Aware Sampling Method ===")
        print(f"Minimum distance threshold: {min_distance}")
        print(f"Distance metric: {distance_metric}")
        
        retain_states = []
        retain_actions = []
        
        # Normalize forget states if requested
        forget_states_normalized = self.forget_states.copy()
        if normalize_states:
            forget_mean = np.mean(self.forget_states, axis=0)
            forget_std = np.std(self.forget_states, axis=0) + 1e-8
            forget_states_normalized = (self.forget_states - forget_mean) / forget_std
        
        total_states_seen = 0
        total_valid_states = 0
        distances_tracked = []

    
        np.random.seed(42)
        self.env.action_space.seed(42)
        seed = 0

        
        for traj_idx in range(n_trajectories):
            self.env.venv.seed(seed)
            state = self.env.reset()[0]
            seed += 1
            trajectory_valid_states = 0
            
            for step in range(max_steps):
                total_states_seen += 1
                
                # Normalize state if needed
                state_normalized = state.copy()
                if normalize_states:
                    state_normalized = (state - forget_mean) / forget_std
                
                # Compute distance to all forget states
                if distance_metric == 'euclidean':
                    distances = np.linalg.norm(
                        forget_states_normalized - state_normalized.reshape(1, -1),
                        axis=1
                    )
                elif distance_metric == 'cosine':
                    # Cosine distance
                    dot_products = np.dot(forget_states_normalized, state_normalized)
                    norms = np.linalg.norm(forget_states_normalized, axis=1) * np.linalg.norm(state_normalized)
                    similarities = dot_products / (norms + 1e-8)
                    distances = 1 - similarities
                elif distance_metric == 'manhattan':
                    distances = np.sum(
                        np.abs(forget_states_normalized - state_normalized.reshape(1, -1)),
                        axis=1
                    )
                else:
                    raise ValueError(f"Unknown distance metric: {distance_metric}")
                
                min_dist = np.min(distances)
                distances_tracked.append(min_dist)
                
                # Check if state is valid (far enough from all forget states)
                if min_dist >= min_distance:
                    # Get action from policy
                    action, _ = self.agent.predict(state, deterministic=deterministic)
                    
                    # Store state and action
                    retain_states.append(state.copy())
                    retain_actions.append(action.copy() if isinstance(action, np.ndarray) else action[0])
                    
                    total_valid_states += 1
                    trajectory_valid_states += 1
                
                # Choose next action (exploration vs exploitation)
                if np.random.random() < random_action_prob:
                    action = self.env.action_space.sample()
                else:
                    action= self.agent.predict(state, deterministic=deterministic)
                
                # Step environment
                if len(action) !=2:
                    action = (action, None)
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state[0]
                
                if done:
                    break
            
            if (traj_idx + 1) % 5 == 0:
                acceptance_rate = (total_valid_states / total_states_seen) * 100 if total_states_seen > 0 else 0
                print(f"Trajectory {traj_idx+1}/{n_trajectories} - "
                      f"Valid states: {total_valid_states}/{total_states_seen} "
                      f"({acceptance_rate:.1f}%) - "
                      f"This trajectory: {trajectory_valid_states} states")
        
        # Convert to arrays
        retain_states = np.array(retain_states) if retain_states else np.empty((0, self.env.observation_space.shape[0]))
        retain_actions = np.array(retain_actions) if retain_actions else np.empty((0, self.env.action_space.shape[0]))
        
        # Print statistics
        print(f"\n=== Distance-Aware Sampling Results ===")
        print(f"Total states seen: {total_states_seen}")
        print(f"Valid states collected: {total_valid_states}")
        print(f"Acceptance rate: {(total_valid_states/total_states_seen)*100:.1f}%")
        print(f"Average min distance: {np.mean(distances_tracked):.3f}")
        print(f"Median min distance: {np.median(distances_tracked):.3f}")
        
        return retain_states, retain_actions

