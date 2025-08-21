# My Test Selector

This project implements a **machine learning test selector** for self-driving car (SDC) systems.  
It extracts geometric and statistical features from road data and uses **multiple ML models** to predict failure probability and select the most valuable test cases.  
The tool runs as a **gRPC service** inside a Docker container.

## How it works
- **Feature extraction** (15 features per road):
  - Basic geometry: total distance, number of points, average segment length, distance variation  
  - Curvature: max/avg/variance of curvature, sharpest turn, average turn angle  
  - Complexity: sharp turns, very sharp turns, turn density  
  - Sequence metrics: direction changes, direction change rate  
  - Statistical: x/y spread of road points  
- **Models used**:
  1. Logistic Regression  
  2. Random Forest  
  3. SVM
  4. Ensemble approach – selects the best model via cross-validation  
- **Training & evaluation**:
  - Historical test outcomes are used for training  
  - Models are compared with 5-fold cross-validation (F1-score)  
  - The best model is selected and trained on the full dataset  
- **Selection strategy**:
  - High probability (>0.7) → always selected  
  - Medium probability (0.4–0.7) → 60% selected  
  - Low probability (<0.4) → 15% selected (for diversity)  


## Requirements
- Docker installed  
- Python dependencies (handled inside Docker):  
  - `numpy`, `scikit-learn`, `grpcio`, etc.  


## Build and Run

### 1. Build Docker image and Run the container
```bash
docker build -t my-selector-image .
docker run --rm --name my-selector-container -t -p 4545:4545 my-selector-image -p 4545
```
