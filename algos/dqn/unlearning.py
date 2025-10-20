import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from typing import Tuple, Dict

def unlearn_policy_kl(agent, forget_states, forget_actions, retain_states=None, retain_actions=None,
                                method='proportional', temperature=100.0, lambda1=0.1, lambda2=0.05,
                                num_steps=50, lr=0.0001, lr_decay=0.95, min_lr=1e-6, 
                                max_grad_norm=1.0, clip_q_values=True, q_scale_factor=0.001,
                                evaluate=True, device=None, debug=False):
    """
    Unlearn specific state-action pairs using a pure full batch approach for maximum stability
    
    Args:
        agent: The CQL agent to modify
        forget_states: Image states to forget
        forget_actions: Actions to forget for these states
        retain_states: Image states to retain (optional)
        retain_actions: Actions for retain states (optional)
        method: Action redistribution method ('proportional', 'uniform')
        temperature: Temperature parameter for softmax conversion
        lambda1: Weight for forget loss
        lambda2: Weight for retain loss
        num_steps: Number of training steps
        lr: Initial learning rate
        lr_decay: Learning rate decay per step
        min_lr: Minimum learning rate
        max_grad_norm: Maximum norm for gradient clipping
        clip_q_values: Whether to scale down Q-values for numerical stability
        q_scale_factor: Factor to scale Q-values if clip_q_values is True
        evaluate: Whether to evaluate unlearning after training
        device: Device to use (if None, will auto-detect)
        debug: Whether to print debug information
        
    Returns:
        agent: The modified agent
        metrics: Evaluation metrics (if evaluate=True)
    """
    # Determine device
    if device is None:
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Access the Q-function based on agent type
    if hasattr(agent, "policy") and hasattr(agent.policy, "q_net"):
        # For DQN-like algorithms
        q_function = agent.policy.q_net
        print(f"Using Q-function from DQN-like agent: {type(q_function).__name__}")
    else:
        print("Warning: Unable to identify Q-function in Stable Baselines agent")
        return agent, {"error": "No accessible Q-function found"}
    
    # Set to evaluation mode
    q_function.eval()
    
    # Enable training for the parameters
    for param in q_function.parameters():
        param.requires_grad = True
    
    # Helper function to extract q_values safely
    def extract_q_values(output):
        if hasattr(output, 'q_value'):
            q_values = output.q_value
        else:
            q_values = output
            
        # Optionally scale down large Q-values for numerical stability
        if clip_q_values:
            q_values = q_values * q_scale_factor
            
        return q_values
    
    # Helper function to compute target distributions
    def compute_target_distributions(probs, actions, method='proportional'):
        target_probs = torch.zeros_like(probs)
        
        for i, (p, action) in enumerate(zip(probs, actions)):
            if method == 'proportional':
                
                mass_to_redistribute = p[action].item()
                num_remaining_actions = len(p) - 1
                mass_per_action = mass_to_redistribute / num_remaining_actions
                target_probs[i] = p.clone()
                target_probs[i][action] = 0
                for j in range(len(p)):
                    if j != action:
                        target_probs[i][j] += mass_per_action
            
            elif method == 'uniform':
                # Uniform redistribution to all other actions
                uniform_probs = torch.ones_like(p)
                uniform_probs[action] = 0
                target_probs[i] = uniform_probs / torch.sum(uniform_probs)
        
        return target_probs
    
    # Convert to tensors - all at once for full batch processing
    forget_states_tensor = torch.tensor(forget_states, dtype=torch.float32, device=device)
    forget_actions_tensor = torch.tensor(forget_actions, dtype=torch.long, device=device)
    
    # Pre-compute original probabilities for retain states (if any)
    if retain_states is not None:
        print("Pre-computing original probabilities for retain states...")
        retain_states_tensor = torch.tensor(retain_states, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            retain_q_output = q_function(retain_states_tensor)
            retain_q_values = extract_q_values(retain_q_output)
            original_retain_probs = F.softmax(retain_q_values / temperature, dim=1)
            print(f"Computed original probabilities for {len(original_retain_probs)} retain states")
    else:
        retain_states_tensor = None
        original_retain_probs = None
    
    # Setup optimizer
    optimizer = torch.optim.Adam(q_function.parameters(), lr=lr, eps=1e-5)  # Increased epsilon for stability
    
    print(f"Starting pure full batch unlearning with {len(forget_states)} forget states")
    if retain_states is not None:
        print(f"and {len(retain_states)} retain states")
    print(f"Using {method} redistribution method with temperature {temperature}")
    print(f"Learning will run for {num_steps} steps with learning rate {lr}")
    
    q_output = q_function(forget_states_tensor)
    q_values = extract_q_values(q_output)

    probs = F.softmax(q_values / temperature, dim=1)
            
    # Create target distribution
    target_probs = compute_target_distributions(probs.detach(), forget_actions_tensor, method)

    original_forget_probs = probs


    # Initialize loss tracking
    losses = {'total': [], 'forget': [], 'retain': []}
    
    # Main training loop
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Process all forget states at once
        try:
            # Forward pass for forget states
            q_output = q_function(forget_states_tensor)
            q_values = extract_q_values(q_output)
                
                # Check for NaN or infinity
            if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                if debug:
                    print(f"Warning: NaN or Inf in Q-values on step {step+1}, skipping")
                continue
                
                
            # Compute KL divergence using log_softmax for better numerical stability
            log_probs = F.log_softmax(q_values / temperature, dim=1)
            forget_loss = F.kl_div(log_probs, target_probs, reduction='batchmean')
            
            # Compute retain loss if retain states are available
            retain_loss = torch.tensor(0.0, device=device)
            
            if retain_states_tensor is not None and original_retain_probs is not None and step % 1 == 0:
                # Forward pass for retain states
                
                retain_q_output = q_function(retain_states_tensor)
                retain_q_values = extract_q_values(retain_q_output)
                    
                # Check for NaN or infinity
                if not (torch.isnan(retain_q_values).any() or torch.isinf(retain_q_values).any()):
                    # Convert to probabilities
                        
                    current_probs = F.log_softmax(retain_q_values / temperature, dim=1)
                        
                    # Compute MSE loss
                    
                    retain_loss = F.kl_div(current_probs, original_retain_probs, reduction='batchmean')
            
            # Compute total loss with weights
            total_loss = lambda1 * forget_loss + lambda2 * retain_loss
            
            # Back propagation
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(q_function.parameters(), max_grad_norm)
            
            # Check for NaN or Inf in gradients
            has_nan_grad = False
            for name, param in q_function.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_grad = True
                        if debug:
                            print(f"Warning: NaN or Inf in gradients for {name}")
            
            if not has_nan_grad:
                # Update parameters
                optimizer.step()
            else:
                print(f"Skipping parameter update on step {step+1} due to NaN/Inf in gradients")
            
            # Store losses
            losses['forget'].append(forget_loss.item())
            losses['retain'].append(retain_loss.item())
            losses['total'].append(total_loss.item())
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            optimizer.param_groups[0]['lr'] = max(current_lr * lr_decay, min_lr)
            
            # Log progress
            if (step + 1) % 1 == 0 or step == 0:
                print(f"step {step+1}/{num_steps}: "
                      f"Total Loss = {total_loss.item():.4f}, "
                      f"Forget Loss = {forget_loss.item():.4f}, "
                      f"Retain Loss = {retain_loss.item():.6f}, "
                      f"LR = {current_lr:.6f}")
                
                # Verify that Q-values are still valid
                if debug:
                    with torch.no_grad():
                        test_states = torch.tensor(forget_states[:3], dtype=torch.float32, device=device)
                        test_q_output = q_function(test_states)
                        test_q_values = extract_q_values(test_q_output)
                        if torch.isnan(test_q_values).any() or torch.isinf(test_q_values).any():
                            print("WARNING: Q-values contain NaN or Inf after update")
                        else:
                            print("Q-values are valid after update")
                            if debug:
                                print(f"Sample Q-values: {test_q_values[0][:3]}...")
                
                del q_output, q_values, log_probs, retain_q_output, retain_q_values, current_probs
                

                    
        except Exception as e:
            print(f"Error on step {step+1}: {e}")
            continue
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(losses['total'], label='Total Loss')
    plt.plot(losses['forget'], label='Forget Loss (KL Divergence)')
    plt.plot(losses['retain'], label='Retain Loss (MSE)')
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    # Evaluate if requested
    if evaluate:
        metrics = evaluate_unlearning(agent, forget_states, forget_actions, 
                                    retain_states, retain_actions, original_retain_probs=original_retain_probs, target_forget_probs=target_probs,
                                    original_forget_probs=original_forget_probs,
                                    temperature=temperature, device=device,
                                    q_scale_factor=q_scale_factor if clip_q_values else 1.0)
        return agent, metrics
    
    return agent, None

def compute_normalized_prob_distance(current_probs, target_probs, original_probs=None):
    """Calculate normalized distances for probability distributions"""
    
    # Standard euclidean distance between probability distributions
    distances = torch.norm(current_probs - target_probs, dim=-1)
    
    if original_probs is not None:
        
        max_possible_distance = torch.norm(original_probs - target_probs, dim=-1)
        
        max_possible_distance = torch.clamp(max_possible_distance, min=1e-10)
        
    else:
        max_possible_distance = np.sqrt(2.0)
    
    normalized_distances = distances / max_possible_distance
    
    return normalized_distances

def evaluate_unlearning(agent, forget_states, forget_actions, retain_states=None, retain_actions=None,
                      original_retain_probs=None, target_forget_probs=None, original_forget_probs=None,
                      temperature=100.0, device=None, q_scale_factor=0.001, distance_threshold=0.5):
    """
    Evaluate the effectiveness of unlearning with pure full batch approach
    
    Args:
        agent: The modified agent
        forget_states: States that should be forgotten
        forget_actions: Actions that should be forgotten
        retain_states: States that should be retained (optional)
        retain_actions: Actions for retain states (optional)
        original_retain_probs: Original probabilities for retain states before unlearning
        target_forget_probs: Target probability distributions for forget states
        target_retain_probs: Target probability distributions for retain states (if different from original)
        temperature: Temperature parameter for softmax
        device: Device to use
        q_scale_factor: Factor to scale Q-values for numerical stability
        distance_threshold: Threshold for distance-based metrics
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    # Try to get the q_function
    if hasattr(agent, "policy") and hasattr(agent.policy, "q_net"):
        q_function = agent.policy.q_net
        
        # Handle ModuleList case
        if isinstance(q_function, torch.nn.ModuleList) and len(q_function) > 0:
            q_function = q_function[0]
    else:
        print("Warning: No direct access to Q-function for evaluation")
        return {}
    
    # Set q_function to evaluation mode
    q_function.eval()
    
    # Helper function to extract q_values safely
    def extract_q_values(output):
        if hasattr(output, 'q_value'):
            q_values = output.q_value
        else:
            q_values = output
            
        # Scale down large Q-values for numerical stability
        q_values = q_values * q_scale_factor
            
        return q_values
    
    # Convert input actions to int explicitly
    forget_actions = np.array(forget_actions).astype(np.int64)
    if retain_actions is not None:
        retain_actions = np.array(retain_actions).astype(np.int64)
    
    # Helper function to get action probabilities
    def get_action_probs(states):
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            try:
                q_output = q_function(states_tensor)
                q_values = extract_q_values(q_output)
                
                # Apply softmax
                probs = F.softmax(q_values / temperature, dim=1)
                return probs
            except Exception as e:
                print(f"Error computing probabilities: {e}")
                # Use zeros as fallback
                action_size = max(forget_actions) + 1 if len(forget_actions) > 0 else 6
                return torch.zeros((len(states), action_size), device=device)
    
    # Get action probabilities
    print("Computing action probabilities for forget states...")
    forget_probs = get_action_probs(forget_states)
    forget_probs_np = forget_probs.cpu().numpy()
    
    # Calculate metrics
    metrics = {}
    
    try:
        # Calculate probability of choosing the forget action
        forget_action_probs = []
        for i, action in enumerate(forget_actions):
            if i < len(forget_probs_np) and action < forget_probs_np.shape[1]:
                forget_action_probs.append(forget_probs_np[i, action])
        
        forget_action_probs = np.array(forget_action_probs)
        
        if len(forget_action_probs) == 0:
            print("Warning: No valid forget action probabilities")
            return {"error": "No valid forget action probabilities"}
        
        # Get probabilities of best alternative actions
        best_alt_action_probs = []
        for i, (probs, action) in enumerate(zip(forget_probs_np, forget_actions)):
            if i >= len(forget_probs_np) or action >= len(probs):
                continue
                
            # Set probability of forget action to -1 to exclude it from argmax
            probs_without_forget = probs.copy()
            probs_without_forget[action] = -1
            
            # Find best alternative action
            best_alt_action = np.argmax(probs_without_forget)
            best_alt_action_probs.append(probs[best_alt_action])
        
        best_alt_action_probs = np.array(best_alt_action_probs)
        
        # Calculate probability ratios
        epsilon = 1e-10  # Avoid division by zero
        prob_ratios = forget_action_probs / (best_alt_action_probs + epsilon)
        
        metrics['forget_action_prob_mean'] = np.mean(forget_action_probs)
        metrics['forget_action_prob_std'] = np.std(forget_action_probs)
        metrics['best_alt_action_prob_mean'] = np.mean(best_alt_action_probs)
        metrics['prob_ratio_mean'] = np.mean(prob_ratios)
        
        # Calculate proportion of states where forget action is no longer the most likely
        best_actions = np.argmax(forget_probs_np, axis=1)
        
        # Ensure indices are integers and valid
        valid_indices = np.minimum(np.arange(len(best_actions)), len(forget_actions)-1).astype(int)
        forget_actions_subset = forget_actions[valid_indices]
        
        # Calculate how many times the best action is different from the forget action
        forget_no_longer_best = np.mean(best_actions != forget_actions_subset)
        metrics['forget_no_longer_best_ratio'] = forget_no_longer_best
        metrics['forget_accuracy'] = forget_no_longer_best * 100  # Percentage where forget action is NOT the most likely
        
        # NEW: Calculate normalized distance to target forget probabilities
        if target_forget_probs is not None:
            #target_forget_probs_tensor = torch.tensor(target_forget_probs, dtype=torch.float32, device=device)
            
            print(original_forget_probs)
            # Compute normalized distances
            forget_distances = compute_normalized_prob_distance(
                forget_probs, 
                target_forget_probs,
                original_probs=original_forget_probs
            )
            
            metrics['forget_mean_distance'] = forget_distances.mean().item()
            metrics['forget_min_distance'] = forget_distances.min().item()
            metrics['forget_max_distance'] = forget_distances.max().item()
            metrics['forget_std_distance'] = forget_distances.std().item()
            
            metrics['forget_total'] = len(forget_distances)
        
        print("\n" + "="*50)
        print("Discrete Action Unlearning Evaluation")
        print("="*50)
        
        print(f"\n📊 Forget States (n={len(forget_actions)}):")
        print(f"  Forget action no longer most likely: {forget_no_longer_best*100:.2f}%")
        print(f"  Average prob of forget actions: {metrics['forget_action_prob_mean']:.4f}")
        print(f"  Average prob of best alternatives: {metrics['best_alt_action_prob_mean']:.4f}")
        print(f"  Average prob ratio (forget/best_alt): {metrics['prob_ratio_mean']:.4f}")
        
        if target_forget_probs is not None:
            print(f"\n  Normalized Distance to Target:")
            print(f"    Mean: {metrics['forget_mean_distance']:.4f} ± {metrics['forget_std_distance']:.4f}")
            print(f"    Range: [{metrics['forget_min_distance']:.4f}, {metrics['forget_max_distance']:.4f}]")
           
        
        # If we have retain states, evaluate those too
        if retain_states is not None and retain_actions is not None:
            print("\n📊 Retain States Evaluation:")
            # Get probabilities for retain states
            retain_probs = get_action_probs(retain_states)
            retain_probs_np = retain_probs.cpu().numpy()
            
            # Calculate probability of choosing the retain action
            retain_action_probs = []
            for i, action in enumerate(retain_actions):
                if i < len(retain_probs_np) and action < retain_probs_np.shape[1]:
                    retain_action_probs.append(retain_probs_np[i, action])
            
            retain_action_probs = np.array(retain_action_probs)
            
            if len(retain_action_probs) > 0:
                metrics['retain_action_prob_mean'] = np.mean(retain_action_probs)
                
                # Calculate proportion of states where retain action is still the most likely
                retain_best_actions = np.argmax(retain_probs_np, axis=1)
                
                # Ensure indices are integers and valid
                valid_retain_indices = np.minimum(np.arange(len(retain_best_actions)), 
                                               len(retain_actions)-1).astype(int)
                retain_actions_subset = retain_actions[valid_retain_indices]
                
                retain_still_best = np.mean(retain_best_actions == retain_actions_subset)
                metrics['retain_still_best_ratio'] = retain_still_best
                metrics['retain_accuracy'] = retain_still_best * 100  # Percentage where retain action IS still the most likely
                
                print(f"  Retain action still most likely: {retain_still_best*100:.2f}%")
                print(f"  Average prob of retain actions: {metrics['retain_action_prob_mean']:.4f}")
                
                # Calculate normalized distance from original probabilities
                if original_retain_probs is not None:
                    #original_retain_probs_tensor = torch.tensor(original_retain_probs, dtype=torch.float32, device=device)
                    
                    # Compute normalized distances
                    retain_distances = compute_normalized_prob_distance(
                        retain_probs,
                        original_retain_probs
                    )
                    
                    metrics['retain_mean_distance'] = retain_distances.mean().item()
                    metrics['retain_min_distance'] = retain_distances.min().item()
                    metrics['retain_max_distance'] = retain_distances.max().item()
                    metrics['retain_std_distance'] = retain_distances.std().item()
                    
                    metrics['retain_total'] = len(retain_distances)
                    
                    print(f"\n  Normalized Distance from Original:")
                    print(f"    Mean: {metrics['retain_mean_distance']:.4f} ± {metrics['retain_std_distance']:.4f}")
                    print(f"    Range: [{metrics['retain_min_distance']:.4f}, {metrics['retain_max_distance']:.4f}]")
                    
                
                # Overall success evaluation
                print("\n" + "="*50)
                
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
    return metrics  


def evaluate(
    agent,
    env,
    n_episodes: int = 20,
    deterministic: bool = True,
    steps: int = None,
) -> Dict[str, float]:
    returns = []
    seed = 0
    for _ in range(n_episodes):
        env.venv.seed(seed)
        obs = env.reset() 
        done = False
        ep_return = 0.0
        step = 0
        while not (done):
            if steps != None and step > steps:
                break
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = env.step(action)
            ep_return += reward
            step += 1
        returns.append(ep_return)
        seed += 1
    returns = np.asarray(returns, dtype=np.float32)
    return {
        "mean":   float(np.mean(returns)),
        "median": float(np.median(returns)),
        "std":    float(np.std(returns, ddof=1)),  
        "all_returns": returns,                    
    }