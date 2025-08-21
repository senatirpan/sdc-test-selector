# My Test Selector

This project implements a **test case selector** for self-driving car (SDC) systems using **road geometry analysis**, **failure probability estimation**, and **clustering for diversity**.  
The tool runs as a **gRPC service** inside a Docker container.

---

## How it works
- **Feature extraction**: From road points, multiple geometric features are computed:
  - Total distance  
  - Number of points  
  - Maximum / average curvature  
  - Curvature standard deviation  
  - Sharp turns  
  - Direction changes  
  - Average segment length  
- **Failure probability**:
  - Estimated using a heuristic model based on road complexity.   
- **Diversity clustering**:
  - Uses K-Means clustering on feature space to ensure variety in selected tests.  
- **Selection strategy**:
  - Prioritizes high failure probability tests  
  - Balances selections across clusters for diversity  
  - Adds some random exploration to avoid bias  

---

## Requirements
- Docker installed  
- Python dependencies (handled inside Docker):  
  - `numpy`, `scikit-learn`, `grpcio`, etc.  

---

## Build and Run

### 1. Build Docker image and Run the container
```bash
docker build -t my-selector-image .

docker run --rm --name my-selector-container -t -p 4545:4545 my-selector-image -p 4545
```
