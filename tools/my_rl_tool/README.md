# My Test Selector

This project implements a **reinforcement learning (RL) test selector** for self-driving car (SDC) systems.  
It applies a **Deep Q-Network (DQN)** to select tests that maximize **fault detection** while maintaining **diversity** in road geometry.  
The tool runs as a **gRPC service** inside a Docker container.

---

## How it works
- **Feature extraction** (15 features):
  - Distance and segment statistics (path length, average/variance)  
  - Curvature (max, avg, std)  
  - Turns and zigzags (sharp turns, very sharp turns, zigzag patterns)  
  - Complexity (direction changes, sinuosity)  
  - Spatial distribution (x/y spread, bounding box ratio)  
- **Reward function**:
  - `+α` reward for detecting a failure  
  - `-β` penalty for selecting similar tests (lack of diversity)  
  - `-γ` selection cost per test  
- **Q-Network**:
  - Input: `[candidate_features, mean_selected_features, min_distance, progress]`  
  - Output: **Q-value** (expected reward of selecting this test)  
  - Architecture: 2 hidden layers (64 units, ReLU), single Q-value output  
- **Training (offline RL)**:
  - Uses historical test outcomes (oracles) to simulate episodes  
  - Stores transitions in replay buffer for experience replay  
  - Trains Q-network with Smooth L1 loss  
  - Soft-updates target network  
- **Selection strategy**:
  - Greedy Q-value selection with epsilon-greedy exploration  
  - Enforces diversity with minimum distance threshold  
  - Selects ~33% of candidates by budget  

---

## Requirements
- Docker installed  
- Python + PyTorch (handled inside Docker)  
- Dependencies: `numpy`, `scikit-learn`, `grpcio`, `torch`  

---

## Build and Run

### 1. Build Docker image and Run the container
```bash
docker build -t rl-selector-image .
docker run --rm --name rl-selector-container -t -p 4545:4545 rl-selector-image -p 4545
```
