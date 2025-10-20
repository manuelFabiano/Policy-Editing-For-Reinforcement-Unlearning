from typing import Optional, List, Tuple, Union, Dict
import numpy as np
import matplotlib.pyplot as plt


class RetainSetGeneratorNoEnv:
    """
    Generate retain-sets without requiring an environment.
    States are sampled uniformly (or via perturbation) in the
    known range [low, high] for each dimension.
    """

    def __init__(
        self,
        agent,
        state_dim: int,
        forget_states: Optional[np.ndarray] = None,
        state_bounds: Optional[List[Tuple[float, float]]] = None,
    ):
        """
        Args
        ----
        agent:        Policy/actor with a .predict(state, deterministic) method
        state_dim:    Dimensionality of the observation space
        forget_states:Optional array of states to forget, shape (k, state_dim)
        state_bounds: List of tuples (low, high) for each state dimension. 
                     If None, defaults to (-8.0, 8.0) for all dimensions.
        """
        self.agent = agent
        self.state_dim = state_dim
        self.forget_states = forget_states
        
        # Set bounds per dimension
        if state_bounds is None:
            self.state_bounds = [(-8.0, 8.0)] * state_dim
        else:
            assert len(state_bounds) == state_dim, \
                f"state_bounds must have {state_dim} elements, got {len(state_bounds)}"
            self.state_bounds = state_bounds

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _sample_uniform_states(self, n: int) -> np.ndarray:
        """Uniform in [low, high] for each dimension."""
        states = np.zeros((n, self.state_dim))
        for i in range(self.state_dim):
            low, high = self.state_bounds[i]
            states[:, i] = np.random.uniform(low, high, size=n)
        return states

    def _actions_from_states(self, states: np.ndarray, deterministic=True) -> np.ndarray:
        """Batch-wise action prediction (falls back to loop if predict is scalar)."""
        try:
            # many SB policies accept a batch already
            actions, _ = self.agent.predict(states, deterministic=deterministic)
        except Exception:
            actions = []
            for s in states:
                a, _ = self.agent.predict(s, deterministic=deterministic)
                actions.append(a)
            actions = np.asarray(actions)
        return actions

    def distance_aware_sampling(
        self,
        n_samples: int = 20_000,
        min_distance: float = 0.5,
        distance_metric: str = "euclidean",
        deterministic: bool = True,
        state_bounds: Optional[List[Tuple[float, float]]] = None,
        max_attempts: int = 1_000_000,
        noise_std: Optional[Union[float, List[float]]] = None,  # scalar or per-dimension
        std_strategy: str = "adaptive",  # "adaptive", "bounds", "forget", "custom"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate retain-states with two strategies:
        - 'uniform' (default): uniform sampling in [low, high]
        - 'perturb': sampling obtained from forget-states + noise 𝒩(0, noise_std²)

        Args:
            std_strategy: Strategy to calculate std per dimension:
                - "adaptive": uses std of forget_states with multiplicative factors
                - "bounds": std = (high - low) * factor for each dimension
                - "forget": std = std(forget_states) for each dimension
                - "custom": uses provided noise_std (must be a list)
            noise_std: If scalar, used as factor. If list, std per dimension.
        """
        if self.forget_states is None:
            raise ValueError("forget_states must be set before sampling")

        # Update bounds if provided
        if state_bounds is not None:
            assert len(state_bounds) == self.state_dim, \
                f"state_bounds must have {self.state_dim} elements, got {len(state_bounds)}"
            self.state_bounds = state_bounds

        # ------------------------------------------------------------------ #
        # 1)  original forget-states and std calculation per dimension
        # ------------------------------------------------------------------ #
        fs = self.forget_states
        
        # Calculate std per dimension based on strategy
        if std_strategy == "custom":
            if noise_std is None or np.isscalar(noise_std):
                raise ValueError("For 'custom' strategy, noise_std must be a list")
            dim_stds = np.array(noise_std)
        elif std_strategy == "forget":
            # Use standard deviation of forget states
            dim_stds = np.std(fs, axis=0)
            dim_stds = np.maximum(dim_stds, 0.1)  # minimum 0.1 to avoid too small std
        elif std_strategy == "bounds":
            # Use a fraction of the bounds range
            factor = noise_std if noise_std is not None and np.isscalar(noise_std) else 0.1
            dim_stds = np.array([(high - low) * factor 
                                for low, high in self.state_bounds])
        elif std_strategy == "adaptive":
            # Combine info from forget_states and bounds
            forget_stds = np.std(fs, axis=0)
            bounds_stds = np.array([(high - low) * 0.1 
                                   for low, high in self.state_bounds])
            # Use the maximum between the two, with a minimum
            dim_stds = np.maximum(np.maximum(forget_stds, bounds_stds), 0.1)
        else:
            raise ValueError(f"Unknown std_strategy: {std_strategy}")
        
        print(f"\nUsing std_strategy='{std_strategy}'")
        print(f"Std per dimension: {dim_stds}")

        # ------------------------------------------------------------------ #
        # 2)  sampling loop
        # ------------------------------------------------------------------ #
        retain_states, distances = [], []
        attempts = 0
        rng = np.random.default_rng()

        while len(retain_states) < n_samples and attempts < max_attempts:
            attempts += 1
            choice = np.random.random()

            # ---- 2.1   generate a candidate ------------------------------- #
            if choice >= 0.8:
                s = self._sample_uniform_states(1)[0]
            else:
                # Adaptive multiplicative factors based on progress
                if len(retain_states) < n_samples // 3:
                    mult_factor = 10.0  # wide exploration at the beginning
                elif len(retain_states) < n_samples // 2:
                    mult_factor = 2.0
                else:
                    mult_factor = 0.5  # more conservative towards the end
                
                base = fs[rng.integers(len(fs))]
                # Noise with dimension-specific std
                noise = rng.normal(0.0, dim_stds * mult_factor)
                s = base + noise
                
                # Clip per dimension using specific bounds
                for i in range(self.state_dim):
                    low, high = self.state_bounds[i]
                    s[i] = np.clip(s[i], low, high)

            # ---- 2.2   distance from original set ------------------------ #
            if distance_metric == "euclidean":
                d = np.linalg.norm(fs - s, axis=1).min()
            elif distance_metric == "cosine":
                dots = fs @ s
                d = (1 - dots /
                    (np.linalg.norm(fs, axis=1) * np.linalg.norm(s) + 1e-8)
                    ).min()
            elif distance_metric == "manhattan":
                d = np.abs(fs - s).sum(1).min()
            else:
                raise ValueError(f"Unknown distance metric {distance_metric}")

            # ---- 2.3   accept or reject ---------------------------------- #
            if d >= min_distance:
                retain_states.append(s)
                distances.append(d)

            if attempts % 50_000 == 0:
                print(f"{attempts:,} attempts – retained {len(retain_states)} "
                    f"({100*len(retain_states)/attempts:.3f} %)")

        # ------------------------------------------------------------------ #
        # 3)  actions + statistics
        # ------------------------------------------------------------------ #
        retain_states  = np.asarray(retain_states)
        retain_actions = self._actions_from_states(retain_states, deterministic)

        print("\n[DistanceAware]")
        print(f"Attempts: {attempts:,}")
        print(f"Retained: {len(retain_states)} / {attempts} "
            f"({100*len(retain_states)/attempts:.3f} %)")
        if distances:
            print(f"avg dist = {np.mean(distances):.3f} | "
                f"median = {np.median(distances):.3f}")

            plt.figure(figsize=(8, 5))
            plt.hist(distances, bins=40, edgecolor="black", alpha=0.7)
            plt.axvline(min_distance, linestyle="--", color="red",
                        label=f"threshold={min_distance}")
            plt.xlabel("min distance to forget-states")
            plt.ylabel("count")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return retain_states, retain_actions

    def analyze_forget_states(self) -> Dict[str, np.ndarray]:
        """
        Analyze forget_states to provide useful statistics for sampling.
        
        Returns:
            Dict with:
            - 'means': mean per dimension
            - 'stds': standard deviation per dimension
            - 'ranges': actual (min, max) per dimension
            - 'suggested_stds': suggested stds for sampling
        """
        if self.forget_states is None:
            raise ValueError("forget_states must be set first")
        
        fs = self.forget_states
        
        # Basic statistics
        means = np.mean(fs, axis=0)
        stds = np.std(fs, axis=0)
        mins = np.min(fs, axis=0)
        maxs = np.max(fs, axis=0)
        ranges = list(zip(mins, maxs))
        
        # Calculate suggested stds combining different info
        forget_stds = stds
        bounds_stds = np.array([(high - low) * 0.1 
                               for low, high in self.state_bounds])
        suggested_stds = np.maximum(np.maximum(forget_stds, bounds_stds), 0.1)
        
        # Print report
        print("\n=== Forget States Analysis ===")
        print(f"Number of forget states: {len(fs)}")
        print("\nPer-dimension statistics:")
        for i in range(self.state_dim):
            print(f"\nDim {i}:")
            print(f"  Range in data: [{mins[i]:.3f}, {maxs[i]:.3f}]")
            print(f"  Bounds:        [{self.state_bounds[i][0]:.3f}, {self.state_bounds[i][1]:.3f}]")
            print(f"  Mean ± Std:    {means[i]:.3f} ± {stds[i]:.3f}")
            print(f"  Suggested std: {suggested_stds[i]:.3f}")
        
        return {
            'means': means,
            'stds': stds,
            'ranges': ranges,
            'suggested_stds': suggested_stds
        }

    # --------------------------------------------------------------------- #
    # Utility
    # --------------------------------------------------------------------- #
    def set_forget_states(self, forget_states: np.ndarray):
        self.forget_states = forget_states
        print(f"Set {len(forget_states)} forget states.")