import numpy as np
import random
from scipy.optimize import linprog

class MDP:
    def __init__(self, num_states=10, num_actions=2, gamma=0.95):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        
        # Randomly generate initial state distribution
        self.mu_0 = np.random.rand(num_states)
        self.mu_0 = self.mu_0 / np.sum(self.mu_0)  # Normalize
        
        # Randomly generate transition probabilities
        self.P = np.zeros((num_states, num_actions, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                self.P[s, a] = np.random.rand(num_states)
                self.P[s, a] = self.P[s, a] / np.sum(self.P[s, a])  # Normalize
        
        # Set true reward function
        self.true_rewards = np.zeros((num_states, num_actions))
        self.true_rewards[:, 0] = 1.0  
        self.true_rewards[:, 1] = 0.9  
        
    def get_optimal_policy(self, rewards=None):
        """
        Compute optimal policy using value iteration
        """
        if rewards is None:
            rewards = self.true_rewards
            
        # Flatten rewards into (state, action) vector
        r_flat = rewards.reshape(-1)
        
        # Construct linear programming matrix - using the LP method from the paper
        A = np.zeros((self.num_states * self.num_actions, self.num_states))
        b = np.zeros(self.num_states * self.num_actions)
        
        for s in range(self.num_states):
            for a in range(self.num_actions):
                idx = s * self.num_actions + a
                A[idx, s] = 1
                for s_next in range(self.num_states):
                    A[idx, s_next] -= self.gamma * self.P[s, a, s_next]
                b[idx] = r_flat[idx]
        
        # Solve linear programming problem for value function
        c = np.zeros(self.num_states)
        for s in range(self.num_states):
            c[s] = (1 - self.gamma) * self.mu_0[s]
        
        res = linprog(c, A_ub=-A, b_ub=-b, method='highs')
        v = res.x
        
        # Calculate optimal policy from value function
        policy = np.zeros((self.num_states, self.num_actions))
        q_values = np.zeros((self.num_states, self.num_actions))
        
        for s in range(self.num_states):
            for a in range(self.num_actions):
                q_values[s, a] = rewards[s, a]
                for s_next in range(self.num_states):
                    q_values[s, a] += self.gamma * self.P[s, a, s_next] * v[s_next]
            
            best_action = np.argmax(q_values[s])
            policy[s, best_action] = 1.0
        
        return policy
    
    def get_occupancy_measure(self, policy):
        """
        Calculate the occupancy measure d_π for policy π
        """
        # Construct state transition matrix P_π
        P_pi = np.zeros((self.num_states, self.num_states))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                P_pi[s] += policy[s, a] * self.P[s, a]
        
        # Solve linear equation (I - γ*P_π)d = (1-γ)μ_0 to get state visitation distribution
        I = np.eye(self.num_states)
        A = I - self.gamma * P_pi
        b = (1 - self.gamma) * self.mu_0
        state_dist = np.linalg.solve(A, b)
        
        # Calculate state-action occupancy measure
        occupancy = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                occupancy[s, a] = state_dist[s] * policy[s, a]
        
        return occupancy
        
    def generate_expert_policy(self):
        """
        Generate expert policy πe = 0.52 × π* + 0.48 × πr
        """
        # Get true optimal policy
        optimal_policy = self.get_optimal_policy()
        
        # Create a greedy suboptimal policy
        suboptimal_policy = np.zeros_like(optimal_policy)
        for s in range(self.num_states):
            best_action = np.argmax(optimal_policy[s])
            suboptimal_action = 1 - best_action  # Choose the other action in binary action space
            suboptimal_policy[s, suboptimal_action] = 1.0
        
        # Mix the two policies
        expert_policy = 0.52 * optimal_policy + 0.48 * suboptimal_policy
        
        # Ensure sum of probabilities in each row is 1
        for s in range(self.num_states):
            expert_policy[s] = expert_policy[s] / np.sum(expert_policy[s])
            
        return expert_policy
