import numpy as np
import scipy.optimize as sopt
from itertools import product
import random

np.random.seed(0)
class LPIRL:
    """
    Linear Programming based offline Inverse Reinforcement Learning algorithm.
    This implementation is based on the paper "A Unified Linear Programming Framework for Offline Reward Learning"
    """
    
    def __init__(self, mdp, N, H, relaxation_level=None, delta=0.1, B=100):
        """
        Initialize the LP-IRL algorithm.
        
        Parameters:
            mdp: MDP object containing state space, action space, transition probabilities, etc.
            N: Number of expert demonstrations
            H: Horizon length
            relaxation_level: Level of equality constraint relaxation. If None, calculated according to the formula in the paper
            delta: Upper bound of failure probability
            B: Radius of the confidence set
        """
        self.mdp = mdp
        self.epsilon_g = 0.01 / np.sqrt(N)
        self.delta = delta
        self.B = B
        
        self.num_states = mdp.num_states
        self.num_actions = mdp.num_actions
        self.gamma = mdp.gamma
        
        # If relaxation level is not specified, calculate according to formula in the paper
        if relaxation_level is None:
            self.epsilon_x = self._compute_relaxation_level(N, H)
        else:
            self.epsilon_x = relaxation_level
            
        # Initialize X matrix (all possible sign vectors)
        self._initialize_X_matrix()
    
    def _compute_relaxation_level(self, N, H):
        """
        Calculate the relaxation level according to the formula in the paper
        """
        self._initialize_X_matrix()
        epsilon_x = (self.B * (1 + self.gamma) * self.gamma**H + 
                     self.B * (1 + self.gamma) * (1 - self.gamma**H)) * \
                    np.sqrt(2 * self.num_states * self.num_actions / N * 
                            np.log(2 * self.X_dim / self.delta))
        
        return epsilon_x
    
    def _initialize_X_matrix(self):
        """
        Initialize X matrix, which contains all possible sign vectors [±1, ±1, ..., ±1]
        """
        # If the state space is large, using all possible sign vectors may cause memory issues
        # In practical applications, we may need to use sampling or other methods to reduce dimensions
        
        # Calculate the number of possible sign vectors
        self.X_dim = 2**self.num_states
        
        # If the state space is large, use sampling or limit the dimension of X
        if self.num_states > 10:  # If states exceed 10, use sampling
            # Randomly sample 100 sign vectors
            X = np.random.choice([-1, 1], size=(self.num_states, 100))
            self.X = X
        else:
            # Generate all possible sign vectors
            signs = list(product([-1, 1], repeat=self.num_states))
            X = np.array(signs).T
            self.X = X
    
    def estimate_occupancy_measure(self, trajectories):
        """
        Estimate occupancy measure from trajectories
        
        Parameters:
            trajectories: List of trajectories, each in the form [(s_0, a_0), (s_1, a_1), ..., (s_H, None)]
            
        Returns:
            de_hat, de_prime_hat: Estimated expert occupancy measure and transition frequencies
        """
        de_hat = np.zeros((self.num_states, self.num_actions))
        de_prime_hat = np.zeros((self.num_states, self.num_actions, self.num_states))
    
        N = len(trajectories)
        for trajectory in trajectories:
            for h, (s, a) in enumerate(trajectory[:-1]):  # excluding the last state
                if a is not None:  # Check if action is valid
                    de_hat[s, a] += self.gamma**h
                
                    # Get the next state
                    s_next, _ = trajectory[h+1]
                    de_prime_hat[s, a, s_next] += self.gamma**h
    
        # Normalize
        de_hat *= (1 - self.gamma) / N
        de_prime_hat *= (1 - self.gamma) / N

        return de_hat, de_prime_hat
    
    def construct_K_matrix(self, de_hat, de_prime_hat):
        """
        Construct the K_D matrix
        
        Parameters:
            de_hat: Estimated expert occupancy measure
            de_prime_hat: Estimated expert state-action-next-state transition frequencies
            
        Returns:
            K_D: Constructed K_D matrix
        """
        K_D = np.zeros((self.num_states, self.num_states * self.num_actions))
        
        for s_prime in range(self.num_states):
            for s in range(self.num_states):
                for a in range(self.num_actions):
                    idx = s * self.num_actions + a
                    if s == s_prime:
                        K_D[s_prime, idx] = de_hat[s, a]
                    K_D[s_prime, idx] -= self.gamma * de_prime_hat[s, a, s_prime]
        
        return K_D
    
    def solve_irl(self, expert_trajectories, suboptimal_trajectories=None):
        """
        Solve the offline IRL problem
        
        Parameters:
            expert_trajectories: List of expert trajectories
            suboptimal_trajectories: List of suboptimal trajectories, used to solve degeneration problems
            
        Returns:
            r: Estimated reward function
        """
        # Estimate expert occupancy measure
        de_hat, de_prime_hat = self.estimate_occupancy_measure(expert_trajectories)
        
        # Construct K_D matrix
        K_D = self.construct_K_matrix(de_hat, de_prime_hat)
        
        # If suboptimal trajectories are provided, estimate suboptimal occupancy measure
        if suboptimal_trajectories is not None:
            dsub_hat, _ = self.estimate_occupancy_measure(suboptimal_trajectories)
        else:
            # If no suboptimal trajectories, use uniform policy as suboptimal policy
            dsub_hat = np.ones((self.num_states, self.num_actions)) / self.num_actions
            # Ensure dsub_hat is a valid occupancy measure (this is just a simplified approximation)
            dsub_hat = dsub_hat / np.sum(dsub_hat) * np.sum(de_hat)
        
        # Define variable dimensions
        n_states = self.num_states
        n_actions = self.num_actions
        n_sa = n_states * n_actions
        n_X = self.X.shape[1]
        
        # Flatten de_hat and dsub_hat
        de_hat_flat = de_hat.reshape(-1)
        dsub_hat_flat = dsub_hat.reshape(-1)
        
        # Construct linear program
        # Variable order: [u (n_sa), v (n_X)]
        n_vars = n_sa + n_X
        
        # Objective function coefficients: maximize u^T(de_hat - dsub_hat)/de_hat
        c = np.zeros(n_vars)
        for i in range(n_sa):
            if de_hat_flat[i] > 0:
                c[i] = 1 - dsub_hat_flat[i] / de_hat_flat[i]
            else:
                c[i] = 0

        
        # Construct constraint matrix
        # Constraint 1: (1-gamma)mu_0^T X v + epsilon_x^T v - u^T 1 <= epsilon_g
        # Constraint 2: K_D^T X v >= u
        # Constraint 3: -de_hat <= u <= de_hat
        # Constraint 4: v >= 0, (range constraint for r is implicit in u's constraint)
        
        # Calculate dimensions of constraint matrix A and right-hand constant b
        # A_1: 1 x n_vars, b_1: 1
        # A_2: n_sa x n_vars, b_2: n_sa
        # A_3 & A_4: 2*n_sa x n_vars, b_3 & b_4: 2*n_sa
        # A_5: n_X x n_vars, b_5: n_X
        
        # Initialize A and b
        n_constraints = 1 + n_sa + 2*n_sa + n_X
        A = np.zeros((n_constraints, n_vars))
        b = np.zeros(n_constraints)
        
        # Constraint 1: (1-gamma)mu_0^T X v + epsilon_x^T v - u^T 1 <= epsilon_g
        # Assume mu_0 is uniform distribution
        mu_0 = np.ones(n_states) / n_states
        A[0, 0:n_sa] = -1  # -u^T 1
        A[0, n_sa:] = (1-self.gamma) * mu_0.dot(self.X) + self.epsilon_x
        b[0] = self.epsilon_g
        
        # Constraint 2: K_D^T X v >= u
        # Equivalent to -K_D^T X v + u <= 0
        for i in range(n_sa):
            A[1+i, i] = 1  # u
            A[1+i, n_sa:] = -K_D.T.dot(self.X)[i, :]
            b[1+i] = 0
        
        # Constraint 3 & 4: -de_hat <= u <= de_hat
        # u <= de_hat
        for i in range(n_sa):
            A[1+n_sa+i, i] = 1
            b[1+n_sa+i] = de_hat_flat[i]
        
        # -de_hat <= u
        # Equivalent to u >= -de_hat
        # Equivalent to -u <= de_hat
        for i in range(n_sa):
            A[1+n_sa+n_sa+i, i] = -1
            b[1+n_sa+n_sa+i] = de_hat_flat[i]
        
        # Constraint 5: v >= 0
        # Equivalent to -v <= 0
        for i in range(n_X):
            A[1+n_sa+2*n_sa+i, n_sa+i] = -1
            b[1+n_sa+2*n_sa+i] = 0
        
        # Constraint 6: u = de_hat ◦ r
        # This requires equality constraints, but linear programs typically use inequality constraints
        # We can add two inequalities to represent equality: u <= de_hat ◦ r and u >= de_hat ◦ r
        # But considering the complexity of the problem, we simplify the handling here
        
        # Solve linear program
        bounds = [(None, None) for _ in range(n_vars)]  # Default variables are unbounded
        
        # Use scipy's linear program solver
        result = sopt.linprog(-c,  # Maximize, so take negative
                             A_ub=A, b_ub=b,
                             bounds=bounds,
                             method='highs')  # Use high-performance solver
        
        if not result.success:
            print("Linear program solving failed:", result.message)
            return None
        
        # Extract reward function
        
        u_flat = result.x[0:n_sa]
        v_flat = result.x[n_sa:]
        
        # Reshape reward function
        r_flat = u_flat / de_hat_flat
        r = r_flat.reshape((self.num_states, self.num_actions))
        
        # Ensure u = de_hat ◦ r
        # For cases where de_hat[s,a] = 0, r[s,a] can be arbitrary, we set to 0
        for s in range(self.num_states):
            for a in range(self.num_actions):
                if de_hat[s, a] == 0:
                    r[s, a] = 0
                else:
                    r[s, a] = u_flat[s*self.num_actions+a] / de_hat[s, a]
        
        return r