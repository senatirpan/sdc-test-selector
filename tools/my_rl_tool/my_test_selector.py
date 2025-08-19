import random
import argparse
import os
import competition_pb2_grpc
import competition_pb2
import grpc
import concurrent.futures as fut
import numpy as np
import json
import math
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, deque
import pickle

# RL dependencies
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """
    Deep Q-Network for test selection.
    Input: [candidate_features, mean_selected_features, min_distance_to_selected, selection_progress]
    Output: Q-value (expected reward for selecting this test now)
    """
    def __init__(self, feature_dim):
        super(QNetwork, self).__init__()
        # Input size: candidate_features + mean_selected_features + distance + progress
        input_dim = feature_dim * 2 + 2
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single Q-value output
        )
        
    def forward(self, x):
        return self.network(x).squeeze(-1)


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))
    
    def sample(self, batch_size=128):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), 
                np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)


class MyTestSelector(competition_pb2_grpc.CompetitionToolServicer):
    """
    RL-based intelligent test selector using DQN to optimize for both 
    fault detection and diversity in road geometry
    """
    
    def __init__(self):
        # Feature processing
        self.feature_scaler = StandardScaler()
        self.test_features = {}  # Cache for computed features
        
        # Historical data for training
        self.historical_data = {}  # testId -> failure boolean
        
        # RL components
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.replay_buffer = ReplayBuffer(50000)
        
        # Training hyperparameters
        self.alpha = 1.0      # Failure detection reward weight
        self.beta = 0.4       # Diversity penalty weight
        self.gamma_r = 0.01   # Selection cost penalty
        self.tau = 1.2        # Distance penalty temperature
        self.gamma = 0.95     # RL discount factor
        self.epsilon = 0.1    # Exploration rate
        
        # Selection parameters
        self.budget_ratio = 0.33  # Select ~33% of tests
        self.min_distance_threshold = 1.0  # Minimum distance for diversity
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialization_complete = False
        
    def Name(self, request, context):
        return competition_pb2.NameReply(name="RL_DQN_TestSelector")
    
    def extract_features(self, road_points):
        """
        Extract comprehensive road geometry features that correlate with SDC failures.
        Returns 15-dimensional feature vector covering distance, curvature, complexity.
        """
        if not road_points or len(road_points) < 2:
            return np.zeros(15)
            
        points = [(p.x, p.y) for p in road_points]
        
        # 1. Distance and segment analysis
        distances = []
        for i in range(len(points)-1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            distances.append(math.sqrt(dx*dx + dy*dy))
        
        total_distance = sum(distances) if distances else 0
        avg_segment_length = np.mean(distances) if distances else 0
        std_segment_length = np.std(distances) if len(distances) > 1 else 0
        
        # 2. Heading and curvature analysis
        headings = []
        curvatures = []
        
        for i in range(len(points)-1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            if dx != 0 or dy != 0:
                heading = math.atan2(dy, dx)
                headings.append(heading)
        
        # Calculate turning angles (curvature proxy)
        turning_angles = []
        for i in range(1, len(headings)):
            delta_theta = headings[i] - headings[i-1]
            # Normalize to [-pi, pi]
            while delta_theta > math.pi:
                delta_theta -= 2*math.pi
            while delta_theta < -math.pi:
                delta_theta += 2*math.pi
            turning_angles.append(abs(delta_theta))
        
        max_curvature = max(turning_angles) if turning_angles else 0
        avg_curvature = np.mean(turning_angles) if turning_angles else 0
        std_curvature = np.std(turning_angles) if len(turning_angles) > 1 else 0
        
        # 3. Sharp turn analysis
        sharp_turns = sum(1 for angle in turning_angles if angle > math.pi/3)  # > 60°
        very_sharp_turns = sum(1 for angle in turning_angles if angle > 5*math.pi/12)  # > 75°
        
        # 4. Direction change complexity
        direction_changes = 0
        if len(turning_angles) > 0:
            signs = [1 if ta > math.pi/6 else -1 if ta < -math.pi/6 else 0 for ta in turning_angles]
            for i in range(1, len(signs)):
                if signs[i] != signs[i-1] and signs[i] != 0 and signs[i-1] != 0:
                    direction_changes += 1
        
        direction_change_density = direction_changes / len(points) if len(points) > 0 else 0
        
        # 5. Zigzag pattern detection
        zigzag_count = 0
        for i in range(2, len(turning_angles)):
            if (turning_angles[i-2] > math.pi/6 and turning_angles[i-1] > math.pi/6 and 
                turning_angles[i] > math.pi/6):
                zigzag_count += 1
        
        # 6. Sinuosity (path efficiency)
        if len(points) >= 2:
            straight_distance = math.sqrt(
                (points[-1][0] - points[0][0])**2 + (points[-1][1] - points[0][1])**2
            )
            sinuosity = 1 - (straight_distance / total_distance) if total_distance > 0 else 0
        else:
            sinuosity = 0
        
        # 7. Spatial distribution
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_spread = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 0
        y_spread = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 0
        
        # 8. Bounding box aspect ratio
        aspect_ratio = x_spread / y_spread if y_spread > 0 else 1.0
        
        return np.array([
            total_distance,           # 0: Path length
            len(points),             # 1: Number of waypoints
            avg_segment_length,      # 2: Average segment length
            std_segment_length,      # 3: Segment length variation
            max_curvature,          # 4: Maximum turning angle
            avg_curvature,          # 5: Average curvature
            std_curvature,          # 6: Curvature variation
            sharp_turns,            # 7: Number of sharp turns (>60°)
            very_sharp_turns,       # 8: Number of very sharp turns (>75°)
            direction_change_density, # 9: Direction changes per point
            zigzag_count,           # 10: Zigzag pattern count
            sinuosity,              # 11: Path inefficiency measure
            x_spread,               # 12: Horizontal extent
            y_spread,               # 13: Vertical extent
            aspect_ratio            # 14: Bounding box aspect ratio
        ])
    
    def compute_diversity_penalty(self, candidate_features, selected_features_list):
        """
        Compute diversity penalty based on minimum distance to already selected tests.
        Returns penalty in [0, 1] where 1 = very similar, 0 = very different
        """
        if not selected_features_list:
            return 0.0  # No penalty for first selection
        
        # Calculate minimum distance to any selected test
        min_distance = float('inf')
        for selected_features in selected_features_list:
            distance = np.linalg.norm(candidate_features - selected_features)
            min_distance = min(min_distance, distance)
        
        # Convert distance to penalty using exponential decay
        penalty = np.exp(-min_distance / self.tau)
        return float(penalty)
    
    def compute_reward(self, has_failed, candidate_features, selected_features_list):
        """
        Compute reward for selecting a test case.
        Reward = α * failure_reward - β * diversity_penalty - γ * selection_cost
        """
        failure_reward = 1.0 if has_failed else 0.0
        diversity_penalty = self.compute_diversity_penalty(candidate_features, selected_features_list)
        selection_cost = self.gamma_r  # Small constant cost per selection
        
        reward = (self.alpha * failure_reward - 
                 self.beta * diversity_penalty - 
                 selection_cost)
        
        return reward
    
    def create_state_vector(self, candidate_features, mean_selected_features, 
                          min_distance, progress):
        """
        Create state vector for DQN input.
        State = [candidate_features, mean_selected_features, min_distance, progress]
        """
        state = np.concatenate([
            candidate_features,
            mean_selected_features,
            [min_distance, progress]
        ])
        return state.astype(np.float32)
    
    def train_dqn_episode(self, features_matrix, labels, budget):
        """
        Simulate one episode of test selection and train DQN.
        This is offline RL using historical oracle data.
        """
        n_tests = len(features_matrix)
        if n_tests == 0:
            return
        
        # Randomize test order for episode diversity
        indices = np.random.permutation(n_tests).tolist()
        
        # Episode state
        selected_indices = []
        selected_features = []
        remaining_indices = indices.copy()
        
        episode_steps = 0
        max_steps = min(budget, len(remaining_indices))
        
        while len(selected_indices) < max_steps and remaining_indices:
            # Current selection progress
            progress = len(selected_indices) / budget
            
            # Mean of selected features (zero vector if none selected)
            if selected_features:
                mean_selected = np.mean(selected_features, axis=0)
            else:
                mean_selected = np.zeros(features_matrix.shape[1])
            
            # Evaluate all remaining candidates
            candidate_q_values = []
            candidate_states = []
            
            for idx in remaining_indices[:min(64, len(remaining_indices))]:  # Limit for performance
                candidate_feat = features_matrix[idx]
                
                # Calculate minimum distance to selected
                if selected_features:
                    distances = [np.linalg.norm(candidate_feat - sf) for sf in selected_features]
                    min_dist = min(distances)
                else:
                    min_dist = 3.0  # Large initial distance
                
                # Create state vector
                state = self.create_state_vector(candidate_feat, mean_selected, min_dist, progress)
                candidate_states.append((state, idx))
                
                # Get Q-value from network
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    q_val = self.q_network(state_tensor).item()
                    candidate_q_values.append(q_val)
            
            # Epsilon-greedy selection
            if random.random() < self.epsilon:
                choice_idx = random.randint(0, len(candidate_states) - 1)
            else:
                choice_idx = np.argmax(candidate_q_values)
            
            # Get selected test
            selected_state, selected_test_idx = candidate_states[choice_idx]
            selected_features.append(features_matrix[selected_test_idx])
            
            # Compute reward
            has_failed = labels[selected_test_idx]
            reward = self.compute_reward(has_failed, features_matrix[selected_test_idx], 
                                       selected_features[:-1])  # Exclude just-selected
            
            # Compute next state (for target Q-value)
            next_selected_indices = selected_indices + [selected_test_idx]
            next_progress = len(next_selected_indices) / budget
            next_mean_selected = np.mean(selected_features, axis=0)
            
            # Store transition in replay buffer
            # For simplicity, we use the current state and approximate next state
            next_state = self.create_state_vector(
                features_matrix[selected_test_idx], next_mean_selected, 0.0, next_progress
            )
            
            done = len(next_selected_indices) >= budget or len(remaining_indices) <= 1
            
            self.replay_buffer.add(selected_state, 0, reward, next_state, done)
            
            # Update episode state
            selected_indices.append(selected_test_idx)
            remaining_indices.remove(selected_test_idx)
            episode_steps += 1
            
            # Train network if buffer has enough samples
            if len(self.replay_buffer) >= 512:
                self.train_dqn_batch()
    
    def train_dqn_batch(self, batch_size=256):
        """Train DQN with a batch of experiences from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
    
    def update_target_network(self, tau=0.01):
        """Soft update of target network parameters"""
        for target_param, local_param in zip(self.target_network.parameters(), 
                                           self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def Initialize(self, request_iterator, context):
        """
        Initialize the RL system by training DQN on historical oracle data.
        This implements offline RL where we simulate episodes using known outcomes.
        """
        print("Starting RL-based initialization...")
        
        # Collect historical data and extract features
        oracle_count = 0
        failure_count = 0
        
        for oracle in request_iterator:
            test_id = oracle.testCase.testId
            has_failed = oracle.hasFailed
            
            # Store outcome
            self.historical_data[test_id] = has_failed
            if has_failed:
                failure_count += 1
            
            # Extract and cache features
            features = self.extract_features(oracle.testCase.roadPoints)
            self.test_features[test_id] = features
            
            oracle_count += 1
            if oracle_count % 1000 == 0:
                print(f"Processed {oracle_count} oracles...")
        
        print(f"Collected {oracle_count} oracles, {failure_count} failures "
              f"(failure rate: {failure_count/oracle_count:.3f})")
        
        # Prepare training data
        feature_list = list(self.test_features.values())
        label_list = [self.historical_data[tid] for tid in self.test_features.keys()]
        
        if not feature_list:
            print("No training data available!")
            return competition_pb2.InitializationReply(ok=False)
        
        # Standardize features
        features_matrix = np.array(feature_list)
        self.feature_scaler.fit(features_matrix)
        features_matrix = self.feature_scaler.transform(features_matrix)
        labels = np.array(label_list)
        
        print(f"Training with {len(features_matrix)} samples, "
              f"feature dimension: {features_matrix.shape[1]}")
        
        # Initialize DQN networks
        feature_dim = features_matrix.shape[1]
        self.q_network = QNetwork(feature_dim).to(self.device)
        self.target_network = QNetwork(feature_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        
        # Training loop
        budget = max(1, int(len(features_matrix) * self.budget_ratio))
        epochs = 2  # Keep training time reasonable
        
        print(f"Training DQN for {epochs} epochs with budget {budget}...")
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            
            # Run multiple episodes per epoch
            episodes_per_epoch = 2
            for episode in range(episodes_per_epoch):
                self.train_dqn_episode(features_matrix, labels, budget)
                
                # Update target network periodically
                if episode % 2 == 0:
                    self.update_target_network()
            
            # Decay epsilon
            self.epsilon = max(0.05, self.epsilon * 0.9)
            
            print(f"  Epsilon: {self.epsilon:.3f}, Buffer size: {len(self.replay_buffer)}")
        
        print("DQN training completed!")
        self.initialization_complete = True
        return competition_pb2.InitializationReply(ok=True)
    
    def Select(self, request_iterator, context):
        """
        Select tests using trained DQN policy.
        Greedy selection based on Q-values with diversity constraints.
        """
        print("Starting RL-based test selection...")
        
        if not self.initialization_complete:
            print("Warning: RL not properly initialized, using fallback")
            return
        
        # Collect all candidates
        candidates = []
        candidate_features = []
        
        for test_case in request_iterator:
            test_id = test_case.testId
            
            # Extract or retrieve features
            if test_id not in self.test_features:
                features = self.extract_features(test_case.roadPoints)
                self.test_features[test_id] = features
            else:
                features = self.test_features[test_id]
            
            candidates.append(test_id)
            candidate_features.append(features)
        
        if not candidates:
            print("No candidates to select from!")
            return
        
        print(f"Selecting from {len(candidates)} candidates...")
        
        # Standardize features
        features_matrix = self.feature_scaler.transform(np.array(candidate_features))
        
        # Selection loop using trained DQN
        budget = max(1, int(len(candidates) * self.budget_ratio))
        selected_indices = []
        selected_features = []
        
        for selection_step in range(budget):
            if len(selected_indices) >= len(candidates):
                break
            
            progress = selection_step / budget
            mean_selected = (np.mean(selected_features, axis=0) if selected_features 
                           else np.zeros(features_matrix.shape[1]))
            
            best_idx = None
            best_q_value = float('-inf')
            
            # Evaluate remaining candidates
            for i, test_id in enumerate(candidates):
                if i in selected_indices:
                    continue
                
                candidate_feat = features_matrix[i]
                
                # Calculate diversity constraint
                if selected_features:
                    distances = [np.linalg.norm(candidate_feat - sf) for sf in selected_features]
                    min_dist = min(distances)
                    if min_dist < self.min_distance_threshold:
                        continue  # Skip if too similar
                else:
                    min_dist = 3.0
                
                # Create state and get Q-value
                state = self.create_state_vector(candidate_feat, mean_selected, min_dist, progress)
                
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    q_value = self.q_network(state_tensor).item()
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_idx = i
            
            # Select best candidate
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_features.append(features_matrix[best_idx])
                
                # Add small exploration chance in later selections
                if selection_step > budget * 0.7 and random.random() < 0.1:
                    # Randomly pick from top candidates for exploration
                    available = [i for i in range(len(candidates)) if i not in selected_indices]
                    if available:
                        exploration_idx = random.choice(available[:min(10, len(available))])
                        if exploration_idx != best_idx:
                            selected_indices[-1] = exploration_idx
                            selected_features[-1] = features_matrix[exploration_idx]
            else:
                break  # No valid candidates left
        
        print(f"Selected {len(selected_indices)} tests using RL policy")
        
        # Yield selected test IDs
        for idx in selected_indices:
            yield competition_pb2.SelectionReply(testId=candidates[idx])


if __name__ == "__main__":
    print("Starting RL-based SDC Test Selector")
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default="4545")
    args = parser.parse_args()
    
    GRPC_PORT = args.port
    GRPC_URL = "[::]:" + GRPC_PORT
    
    server = grpc.server(fut.ThreadPoolExecutor(max_workers=4))
    competition_pb2_grpc.add_CompetitionToolServicer_to_server(MyTestSelector(), server)
    server.add_insecure_port(GRPC_URL)
    
    print(f"Starting RL server on port {GRPC_PORT}")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    server.start()
    print("RL Test Selector is running...")
    server.wait_for_termination()
    print("Server terminated")