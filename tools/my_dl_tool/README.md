# My Test Selector

This project implements a **self-driving car (SDC) test case selector** using geometric road features and a trained **CNN + BiLSTM hybrid model**.  
The tool runs as a **gRPC server** inside a Docker container and selects test cases based on predicted failure probabilities.

---

## How it works
- **Feature extraction**: From road coordinates, three features are computed:
  - Turning angles
  - Segment lengths
  - Curvature
- **Preprocessing**: Features are padded to fixed sequences of length 50 and scaled with a pre-trained scaler.
- **Prediction**: The ONNX model predicts the failure probability for each test case.
- **Selection**: The selector chooses ~25% of tests, prioritizing those with the highest failure probability.

---

## Requirements
- Docker installed
- Model files available in the project root:
  - `best_model.onnx`
  - `best_model_scaler.pkl`

---

## Build and Run

### 1. Build Docker image and Run the container
```bash
docker build -t my-selector-image .

docker run --rm --name my-selector-container -t -p 4545:4545 my-selector-image -p 4545

```