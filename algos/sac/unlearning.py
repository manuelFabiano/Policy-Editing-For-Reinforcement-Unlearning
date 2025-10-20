import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, Union
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution


def compute_normalized_distance(current_actions, target_actions, action_space, forget_actions=None):
    """Calculate normalized distances with respect to the action range"""
    
    # Standard euclidean distance
    distances = torch.norm(current_actions - target_actions, dim=-1)
    if forget_actions != None: 
        max_possible_distance = torch.norm(forget_actions - target_actions, dim=-1)
    else:
        # Normalize by action range
        action_range = action_space.high - action_space.low
        max_possible_distance = np.linalg.norm(action_range)
        print(max_possible_distance)
    normalized_distances = distances / max_possible_distance
    
    return normalized_distances

def evaluate_unlearning(
    agent,
    forget_states: np.ndarray,
    forget_actions: np.ndarray,
    retain_states: Optional[np.ndarray] = None,
    retain_actions: Optional[np.ndarray] = None,
    target_forget_means: Optional[torch.Tensor | np.ndarray] = None,
    target_retain_means: Optional[torch.Tensor | np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Evaluate the KL-divergence based unlearning with accuracy metrics
    """
    if device is None:
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    metrics = {}
    
    with torch.no_grad():
        # Evaluate forget states
        forget_states_tensor = torch.tensor(forget_states, dtype=torch.float32, device=device)
        forget_actions_tensor = torch.tensor(forget_actions, dtype=torch.float32, device=device)
        
        
        current_forget_actions = agent.policy.actor(forget_states_tensor, deterministic=True)
        
        
        # Compute forget distances
        #forget_distances = torch.norm(current_forget_actions - forget_actions_tensor, dim=-1)
        forget_distances = compute_normalized_distance(current_forget_actions, forget_actions_tensor, agent.action_space)
        
        metrics['forget_mean_distance'] = forget_distances.mean().item()
        metrics['forget_min_distance'] = forget_distances.min().item()
        metrics['forget_max_distance'] = forget_distances.max().item()
        metrics['forget_std_distance'] = forget_distances.std().item()
        
        metrics['forget_total'] = len(forget_distances)

        if target_forget_means is not None:
            #dist_forget_target = torch.norm(current_forget_actions - target_forget_means, dim=-1)
            features = agent.policy.actor.features_extractor(forget_states_tensor)
            latent_pi = agent.policy.actor.latent_pi(features)
            current_forget_means = agent.policy.actor.mu(latent_pi)
            dist_forget_target = compute_normalized_distance(current_forget_means, target_forget_means, agent.action_space, forget_actions=forget_actions_tensor)

            metrics['forget_target_mean_distance'] = dist_forget_target.mean().item()
            metrics['forget_target_min_distance'] = dist_forget_target.min().item()
            metrics['forget_target_max_distance'] = dist_forget_target.max().item()
            metrics['forget_target_std_distance'] = dist_forget_target.std().item()
        
        # Evaluate retain states if provided
        if retain_states is not None:
            retain_states_tensor = torch.tensor(retain_states, dtype=torch.float32, device=device)
            
            # Get current actions for retain states
            current_retain_actions = agent.policy.actor(retain_states_tensor, deterministic=True)
            
            target_retain = torch.tensor(retain_actions, dtype=torch.float32, device=device)
            
            # Compute retain distances
            retain_distances = torch.norm(current_retain_actions - target_retain, dim=-1)
            
            metrics['retain_mean_distance'] = retain_distances.mean().item()
            metrics['retain_min_distance'] = retain_distances.min().item()
            metrics['retain_max_distance'] = retain_distances.max().item()
            metrics['retain_std_distance'] = retain_distances.std().item()
            
            if target_retain_means is not None:
                features = agent.policy.actor.features_extractor(retain_states_tensor)
                latent_pi = agent.policy.actor.latent_pi(features)
                current_retain_means = agent.policy.actor.mu(latent_pi)
                #dist_retain_target = torch.norm(current_retain_means - target_retain_means, dim=-1)
                dist_retain_target = compute_normalized_distance(current_retain_means, target_retain_means, agent.action_space)

                metrics['retain_target_mean_distance'] = dist_retain_target.mean().item()
                metrics['retain_target_min_distance'] = dist_retain_target.min().item()
                metrics['retain_target_max_distance'] = dist_retain_target.max().item()
                metrics['retain_target_std_distance'] = dist_retain_target.std().item()
            
            metrics['retain_total'] = len(retain_distances)
    
    print("\n" + "="*50)
    print("KL-Divergence Unlearning Evaluation")
    print("="*50)
    
    print(f"\n📊 Forget States (n={metrics['forget_total']}):")
    print(f"  Distance to original forget states: {metrics['forget_mean_distance']:.4f} ± {metrics['forget_std_distance']:.4f}")
    print(f"  Range: [{metrics['forget_min_distance']:.4f}, {metrics['forget_max_distance']:.4f}]")
    if target_forget_means is not None:
        print(f"  Distance to target forget actions: {metrics['forget_target_mean_distance']:.4f} ± {metrics['forget_target_std_distance']:.4f}")
    
    if 'retain_mean_distance' in metrics:
        print(f"\n📊 Retain States (n={metrics['retain_total']}):")
        if target_retain_means is not None:
            print(f"  Distance to target retain actions: {metrics['retain_target_mean_distance']:.4f} ± {metrics['retain_target_std_distance']:.4f}")
        # Overall success metrics
        print("\n" + "="*50)
    
    return metrics

def unlearn_continuous_policy_kl(
    agent,
    forget_states: np.ndarray,
    forget_actions: np.ndarray,
    retain_states: Optional[np.ndarray] = None,
    retain_actions: Optional[np.ndarray] = None,
    method: str = 'random',
    temperature: float = 1.0,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    num_steps: int = 100,
    lr: float = 0.0001,
    lr_decay: float = 0.95,
    min_lr: float = 1e-6,
    max_grad_norm: float = 1.0,
    evaluate: bool = True,
    device: Optional[torch.device] = None,
    debug: bool = False
) -> Tuple:
    """
    Unlearn specific state-action pairs using KL divergence for continuous control
    
    This follows the same approach as the discrete version but adapted for continuous actions
    
    Args:
        agent: The continuous control agent (SAC, TD3, PPO)
        forget_states: States to forget
        forget_actions: Actions to forget
        retain_states: States to retain (optional)
        retain_actions: Actions for retain states (optional)
        method: Target distribution method ('uniform', 'gaussian_away', 'opposite')
        temperature: Temperature for distribution sharpness
        lambda1: Weight for forget loss
        lambda2: Weight for retain loss
        num_steps: Number of training steps
        lr: Initial learning rate
        lr_decay: Learning rate decay per step
        min_lr: Minimum learning rate
        max_grad_norm: Maximum norm for gradient clipping
        evaluate: Whether to evaluate after training
        device: Device to use
        debug: Whether to print debug information
        
    Returns:
        agent: Modified agent
        metrics: Evaluation metrics
    """
    # Determine device
    if device is None:
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Identify algorithm type and get appropriate networks
    algorithm_type = None
    actor_network = None
    
    if hasattr(agent, "policy"):
        policy = agent.policy
        
        # Check for SAC (has squashed gaussian actor)
        if hasattr(policy, "actor") and hasattr(policy.actor, "log_std"):
            algorithm_type = "SAC"
            actor_network = policy.actor
            print("Detected SAC agent - using stochastic actor with Gaussian distribution")
    
    if algorithm_type is None:
        raise ValueError("Could not identify agent type (SAC/TD3/PPO)")
    
    # Set to training mode
    actor_network.train()
    
    
    # Enable gradients
    params_to_train = actor_network.parameters()
    for param in params_to_train:
        param.requires_grad = True
    
    # Convert to tensors
    forget_states_tensor = torch.tensor(forget_states, dtype=torch.float32, device=device)
    forget_actions_tensor = torch.tensor(forget_actions, dtype=torch.float32, device=device)
    
    # Function to get action distribution based on algorithm type
    def get_action_distribution(states):
        if algorithm_type == "SAC":
            features = actor_network.features_extractor(states)
            latent_pi = actor_network.latent_pi(features)
            mean = actor_network.mu(latent_pi)
            log_std = actor_network.log_std(latent_pi)
            std = torch.exp(log_std)
            return D.Normal(mean, std), mean
            
    # Pre-compute original distributions for retain states
    if retain_states is not None:
        print("Pre-computing original actions for retain states...")
        retain_states_tensor = torch.tensor(retain_states, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            original_retain_dist, original_mean = get_action_distribution(retain_states_tensor)
            original_retain_mean = original_retain_dist.mean.detach()
    else:
        retain_states_tensor = None
        original_retain_mean = None
    
    # Setup optimizer
    optimizer = torch.optim.Adam(actor_network.parameters(), lr=lr)
    
    
    print(f"\nStarting KL-divergence based unlearning:")
    print(f"Algorithm: {algorithm_type}")
    print(f"Method: {method}")
    print(f"Temperature: {temperature}")
    print(f"Forget states: {len(forget_states)}")
    print(f"Retain states: {len(retain_states) if retain_states is not None else 0}")
    
    # Loss tracking
    losses = {'total': [], 'forget': [], 'retain': []}
    

    # Create target distribution based on method
    if method == 'random':
        target_mean = torch.empty_like(forget_actions_tensor).uniform_(-1, 1)
        target_std = torch.ones_like(forget_actions_tensor) * 0.1 * temperature
        
    elif method == 'opposite':
            # Target is distribution centered at opposite actions
        target_mean = -forget_actions_tensor
        target_std = torch.ones_like(forget_actions_tensor) * 0.1 * temperature
            
    else:
        raise ValueError(f"Unknown method: {method}")
        
    # Create target distribution
    target_dist = D.Normal(target_mean, target_std)
        
    best_loss=10000000000
    k=0
    # Main training loop
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Get current action distribution for forget states
        current_dist, current_mean = get_action_distribution(forget_states_tensor)
        
    
        # Compute KL divergence
        forget_loss = D.kl_divergence(current_dist, target_dist).mean()
           
        # Alternative: sample-based KL estimation if closed form is problematic
        if torch.isnan(forget_loss) or torch.isinf(forget_loss):
            # Sample-based approximation
            samples = current_dist.rsample()  # Reparameterization trick
            current_log_prob = current_dist.log_prob(samples)
            target_log_prob = target_dist.log_prob(samples)
            forget_loss = (current_log_prob - target_log_prob).mean()
        
        # Compute retain loss
        retain_loss = torch.tensor(0.0, device=device)
        if retain_states_tensor is not None and original_retain_mean is not None:
            current_retain_dist, current_retain_mean = get_action_distribution(retain_states_tensor)
            
            retain_loss = D.kl_divergence(current_retain_dist, original_retain_dist).mean()

        # Total loss
        total_loss = lambda1 * forget_loss + lambda2 * retain_loss
        
        if total_loss < best_loss:
            best_loss = total_loss
            k = 0
        else:
            k = k+1 
        if k == 50:
            break

        # Check for NaN
        if torch.isnan(total_loss):
            if debug:
                print(f"NaN detected at step {step}, skipping...")
            continue
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(params_to_train, max_grad_norm)
        
        # Update
        optimizer.step()
        
        # Update learning rate
        if step % 25000==0:
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = max(current_lr * 0.1, min_lr)
        
        # Store losses
        losses['forget'].append(forget_loss.item())
        losses['retain'].append(retain_loss.item())
        losses['total'].append(total_loss.item())
        
        # Logging
        if (step + 1) % 10 == 0 or step == 0:
            print(f"step {step+1}/{num_steps}: "
                  f"Total Loss = {total_loss.item():.4f}, "
                  f"Forget Loss (KL) = {forget_loss.item():.4f}, "
                  f"Retain Loss (KL) = {retain_loss.item():.6f}, "
                  f"LR = {current_lr:.6f}")
            
            if debug:
                # Check action changes
                with torch.no_grad():
                    test_dist = get_action_distribution(forget_states_tensor[:5])
                    test_actions = test_dist.mean
                    distances = torch.norm(test_actions - forget_actions_tensor[:5], dim=-1)
                    print(f"  Sample distances from forget actions: {distances.cpu().numpy()}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(losses['total'], label='Total Loss', linewidth=2)
    plt.plot(losses['forget'], label='Forget Loss (KL)', linewidth=2)
    plt.plot(losses['retain'], label='Retain Loss (MSE)', linewidth=2)
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.title(f'KL-Divergence Unlearning ({algorithm_type})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.show()
    
    # Evaluate if requested
    if evaluate:
        metrics = evaluate_unlearning(
            agent, forget_states, forget_actions,
            retain_states, retain_actions,
            target_forget_means= target_mean,
            target_retain_means=original_mean,
            device=device
        )
        return agent, metrics,target_mean, original_mean
    
    return agent, None, target_mean, original_mean


def evaluate(
    agent,
    env,
    n_episodes: int = 20,
    deterministic: bool = True,
) -> Dict[str, float]:
    returns = []
    seed = 0
    for _ in range(n_episodes):
        obs, _ = env.reset(seed = seed) 
        done = False
        ep_return = 0.0
        
        while not (done):
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            ep_return += reward

        returns.append(ep_return)
        seed += 1
    returns = np.asarray(returns, dtype=np.float32)
    return {
        "mean":   float(np.mean(returns)),
        "median": float(np.median(returns)),
        "std":    float(np.std(returns, ddof=1)),  # sample standard deviation
        "all_returns": returns,                    # useful for extra analysis
    }
