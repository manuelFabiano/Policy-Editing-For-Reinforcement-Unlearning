import torch
import torch.nn as nn
import numpy as np

class RetainSetGenerator:
    """
    Class for estimating the retain set for IQL algorithms
    """
    def __init__(self, agent, environment=None, feature_extractor=None,embedding_dim=512, state_dim=None, forget_states=None, similarity_threshold=None):
        """
        Initialize the estimator
        
        Args:
            agent: The trained RL agent
            feature_extractor: Optional neural network to extract features from images
                              If None, a default feature extractor is created
            embedding_dim: Dimension of the image embeddings (ignored if feature_extractor is provided)
        """

        self.agent = agent
        self.feature_extractor = feature_extractor
        self.env = environment
        self.similarity_threshold = similarity_threshold
       
        self.forget_states = forget_states
        self.policy = None

        # Access the policy and Q-function
        try:
            if hasattr(agent, "_impl"):
                self.policy = agent._impl.policy if hasattr(agent._impl, "policy") else None
                self.q_function = agent._impl.q_function[0] if hasattr(agent._impl, "q_function") else None
            else:
                self.policy = None
                self.q_function = None
        except: pass

        if self.policy is not None:
            self.device = next(self.policy.parameters()).device
        else:
            self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

        # Create a default feature extractor if none is provided
        if self.feature_extractor is None:
            self.feature_extractor = self._create_default_feature_extractor(embedding_dim)
            print(f"Created default feature extractor with output dimension {embedding_dim}")

    
        
        # Set up distance filter if forget states are provided
        if forget_states is not None and similarity_threshold is not None and similarity_threshold > 0:
            print(f"Filtering generated states to maintain min_distance={similarity_threshold} from forget states")
            self.is_valid_state = self._setup_distance_filter(similarity_threshold)
        else:
            # If no filtering required, all states are valid
            self.is_valid_state = lambda state: True
    

    def _create_default_feature_extractor(self, embedding_dim=512):
        """
        Create a simple CNN feature extractor
        
        Args:
            embedding_dim: Dimension of the output embedding
            
        Returns:
            A neural network that converts image observations to embeddings
        """
        # Simple CNN architecture for Atari-like images (4,84,84)
        class FeatureExtractor(nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
                self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
                self.fc1 = nn.Linear(64 * 7 * 7, embedding_dim)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                # Ensure input has the right shape and type and is on the correct device
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32, device=self.fc1.weight.device)
                if x.dim() == 3:  # Single image with shape (84, 84, 4)
                    x = x.unsqueeze(0)  # Add batch dimension -> (1, 84, 84, 4)
                
                # Permute dimensions from (batch, height, width, channels) to (batch, channels, height, width)
                if x.shape[1] == 84 and x.shape[2] == 84 and x.shape[3] == 4:
                    x = x.permute(0, 3, 1, 2).contiguous()  # Reorder to (batch, 4, 84, 84) and make contiguous
                
                # Ensure x is on the same device as the model
                if x.device != self.fc1.weight.device:
                    x = x.to(self.fc1.weight.device)
                
                # Normalize if needed
                if x.max() > 1.0:
                    x = x / 255.0
                
                # Forward pass
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = x.reshape(x.size(0), -1)  # Flatten using reshape instead of view
                x = self.fc1(x)
                return x
        model = FeatureExtractor(embedding_dim)
        
        model = model.to(self.device)    
        # Put in eval mode
        model.eval()
        
        return model
    
    def get_embedding(self, state):
        """
        Convert an image observation to an embedding
        
        Args:
            state: Image observation of shape (4,84,84) or batch of images
            
        Returns:
            Embedding vector(s)
        """
        with torch.no_grad():
            # Ensure state has the right shape
            if isinstance(state, np.ndarray):
                if state.ndim == 3:  # Single image (4,84,84)
                    state = np.expand_dims(state, 0)  # Add batch dimension
            
            # Get embedding
            embedding = self.feature_extractor(state)
            
            # Convert to numpy
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            return embedding
    
    def compute_embeddings_batch(self, states):
        """
        Compute embeddings for a batch of states
        
        Args:
            states: Batch of image observations
            
        Returns:
            Batch of embeddings
        """
        embeddings = []
        batch_size = 32  # Process in batches to avoid memory issues
        
        for i in range(0, len(states), batch_size):
            batch = states[i:i+batch_size]
            batch_embeddings = self.get_embedding(batch)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    

    def get_action(self, state):
        """
        Get action from agent for a given state
        """
        
        
        return self.agent.predict(state, deterministic=True)
    
    
    def _setup_distance_filter(self, similarity_threshold):
        """
        Set up a radius-based nearest neighbors model for filtering image states
        
        Args:
            forget_states: Images to filter against (n_states, C, H, W)
            similarity_threshold: Radius threshold in embedding space
            
        Returns:
            A function that returns True if a state is valid (sufficiently distant in embedding space)
        """
        if self.forget_states is None or len(self.forget_states) == 0 or similarity_threshold is None or similarity_threshold <= 0:
            # If no filtering required, all states are valid
            print("No distance filtering will be applied")
            return lambda state: True
        
        print(f"Setting up distance filter with similarity_threshold={similarity_threshold}")
        print(f"Computing embeddings for {len(self.forget_states)} forget states...")
        
        # Compute embeddings for forget states
        forget_embeddings = self.compute_embeddings_batch(self.forget_states)
        forget_embeddings = forget_embeddings.astype(np.float32)
        
        # Build nearest neighbors model with radius search
        from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
        # Build kNN model instead of radius search (cuML API difference)
        k = 1  # We only need the closest neighbor
        nn_forget = cuNearestNeighbors(n_neighbors=k)
        nn_forget.fit(forget_embeddings)
        
        # Define the filter function
        def distance_filter(state):
            # Get embedding for the state
            embedding = self.get_embedding(state)[0].astype(np.float32)
            
            # Get distance to nearest point in forget set
            distances, _ = nn_forget.kneighbors(embedding.reshape(1, -1))
            
            # Valid if closest point is farther than threshold
            return distances[0][0] >= similarity_threshold
        
        return distance_filter

    
    def estimate_retain_set_via_environment(self, n_trajectories=10, max_steps=100, random_action_prob=0.2):
        """
        Estimate the retain set by rolling out trajectories in the environment
        
        Args:
            env: Gymnasium environment (Atari, with image observations)
            n_trajectories: Number of trajectories to collect
            max_steps: Maximum steps per trajectory
            forget_states: States to exclude (images to forget)
            similarity_threshold: Radius threshold in embedding space
            random_action_prob: Probability of taking random actions for exploration
                
        Returns:
            retain_states: The estimated states for the retain set (images)
            retain_actions: The actions for these states
        """
        env = self.env
        if env is None:
            raise ValueError("Environment is required for this method")
        
        retain_states = []
        retain_actions = []
        
        print("Generating retain set by rolling out trajectories in the environment...")
        
        
        # Track total states collected
        total_valid_states = 0
        total_states_seen = 0
        np.random.seed(42)
        env.action_space.seed(42)
        seed = 0
        for traj_idx in range(n_trajectories):
            print(f"Trajectory {traj_idx+1}/{n_trajectories} - Collected {total_valid_states} valid states out of {total_states_seen} total states")
            
            # Reset the environment to get an initial state
            env.venv.seed(seed)
            state = env.reset()
            seed += 0
            
            for step in range(max_steps):
                total_states_seen += 1
                
                # Check if the state is valid (sufficiently different from forget states)
                if self.is_valid_state(state):
                    # Get the optimal action for this state from the agent
                    action = self.agent.predict(state, deterministic=True)[0]
                    
                    # Add to our results
                    retain_states.append(state.copy())
                
                    
                    retain_actions.append(action)
                    total_valid_states += 1
                
                # Decide whether to take a random action or follow the policy
                if np.random.random() < random_action_prob:
                    # Take a random action for exploration
                    action = env.action_space.sample()
                else:
                    # Use the agent's policy
                    action = self.agent.predict(state, deterministic=True)[0]
                                
                # Step the environment
            
                if not isinstance(action, (list, tuple, np.ndarray)):
                    action = [action]
                next_state, reward, done, _ = env.step(action)
                state = next_state
            
                
                if done:
                    break
        
        # Convert to arrays if we have any states
        if retain_states:
            retain_states = np.array(retain_states)
            retain_actions = np.array(retain_actions)
        
        print(f"Collected {len(retain_states)} valid states out of {total_states_seen} total states")
        print(f"Acceptance rate: {len(retain_states)/total_states_seen:.2%}")
        
        return retain_states, retain_actions


    