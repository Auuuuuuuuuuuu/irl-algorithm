import numpy as np
import random

def sample_trajectory(mdp, policy, horizon=20):
    """
    Sample a trajectory based on the given policy
    
    Parameters:
        mdp: MDP object
        policy: Policy matrix of shape (num_states, num_actions)
        horizon: Length of the trajectory
        
    Returns:
        trajectory: List of (state, action) pairs
    """
    trajectory = []
    # Sample initial state from initial state distribution
    state = np.random.choice(mdp.num_states, p=mdp.mu_0)
    
    for h in range(horizon):
        # Choose action according to policy
        action = np.random.choice(mdp.num_actions, p=policy[state])
        
        # Record state-action pair
        trajectory.append((state, action))
        
        # Transition to next state
        next_state = np.random.choice(mdp.num_states, p=mdp.P[state, action])
        state = next_state
    
    # Add final state
    trajectory.append((state, None))
    
    return trajectory

def generate_irl_dataset(mdp, num_samples, horizon=20):
    """
    Generate IRL dataset
    
    Parameters:
        mdp: MDP object
        num_samples: Number of trajectories to sample
        horizon: Length of each trajectory
        
    Returns:
        trajectories: List of sampled trajectories
        expert_policy: Expert policy used for sampling
    """
    # Get expert policy
    expert_policy = mdp.generate_expert_policy()
    
    # Sample trajectories from expert policy
    trajectories = []
    for _ in range(num_samples):
        traj = sample_trajectory(mdp, expert_policy, horizon)
        trajectories.append(traj)
    
    return trajectories, expert_policy

def generate_human_feedback_dataset(mdp, num_samples, feedback_type='discrete', horizon=20):
    """
    Generate RLHF dataset
    
    Parameters:
        mdp: MDP object
        num_samples: Number of trajectory pairs to sample
        feedback_type: Type of feedback ('discrete' or 'continuous')
        horizon: Length of each trajectory
        
    Returns:
        feedback_data: List of (trajectory1, trajectory2, feedback) tuples
    """
    # Get expert policy and uniform policy
    expert_policy = mdp.generate_expert_policy()
    uniform_policy = np.ones((mdp.num_states, mdp.num_actions)) / mdp.num_actions
    
    # 2/3 of samples from expert policy, 1/3 from uniform policy
    policy = expert_policy if random.random() < 2/3 else uniform_policy
    
    # Generate trajectory pairs and human feedback
    feedback_data = []
    for _ in range(num_samples):
        # Sample two trajectories
        traj1 = sample_trajectory(mdp, policy, horizon)
        traj2 = sample_trajectory(mdp, policy, horizon)
        
        # Calculate cumulative reward for each trajectory
        r1 = compute_trajectory_reward(mdp, traj1)
        r2 = compute_trajectory_reward(mdp, traj2)
        
        # Generate human feedback
        if feedback_type == 'discrete':
            # Generate discrete feedback using BTL model
            p1 = 1 / (1 + np.exp(-(r1 - r2)))  # Probability of trajectory 1 under BTL model
            y = 1 if random.random() < p1 else 2
        else:
            # Generate continuous feedback
            reward_diff = r1 - r2
            # Sample from uniform distribution in [0, 0.2*reward_diff]
            y = random.uniform(0, 0.2 * abs(reward_diff)) * (1 if reward_diff > 0 else -1)
        
        feedback_data.append((traj1, traj2, y))
    
    return feedback_data

def compute_trajectory_reward(mdp, trajectory):
    """
    Calculate cumulative reward of a trajectory
    
    Parameters:
        mdp: MDP object
        trajectory: List of (state, action) pairs
        
    Returns:
        total_reward: Cumulative discounted reward
    """
    total_reward = 0
    for h, (state, action) in enumerate(trajectory[:-1]):  # Excluding the last state
        reward = mdp.true_rewards[state, action]
        total_reward += mdp.gamma**h * reward
    return total_reward