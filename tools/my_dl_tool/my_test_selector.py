import argparse
import os
import json
import numpy as np
import math
import joblib
import onnxruntime as ort
import competition_pb2_grpc
import competition_pb2
import grpc
import concurrent.futures as fut


def calculate_angle(p1, p2, p3):
    """Calculate angle between three points with improved handling"""
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Handle zero vectors
    if np.allclose(v1, 0) or np.allclose(v2, 0):
        return 0
    
    # Calculate angle with numerical stability
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norms == 0:
        return 0
    
    cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    # Convert to degrees and return as turning angle
    angle_deg = np.degrees(angle)
    return 180 - angle_deg


def calculate_curvature(p1, p2, p3):
    """Calculate curvature at point p2"""
    # Using the formula: curvature = |cross_product| / |v|^3
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    if np.allclose(v1, 0) or np.allclose(v2, 0):
        return 0
    
    # Cross product for 2D vectors
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    v_norm = np.linalg.norm(v1 + v2)
    
    if v_norm == 0:
        return 0
    
    return abs(cross) / (v_norm ** 3)


def calculate_segment_length(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def extract_three_features(road_coordinates):
    """Extract three features: angles, lengths, and curvature"""
    if len(road_coordinates) < 3:
        return [], [], []
    
    angles = []
    lengths = []
    curvatures = []
    
    # Calculate segment lengths
    for i in range(len(road_coordinates) - 1):
        length = calculate_segment_length(road_coordinates[i], road_coordinates[i + 1])
        lengths.append(length)
    
    # Calculate angles and curvatures (need at least 3 points)
    for i in range(len(road_coordinates) - 2):
        angle = calculate_angle(road_coordinates[i], road_coordinates[i + 1], road_coordinates[i + 2])
        angles.append(angle)
        
        curvature = calculate_curvature(road_coordinates[i], road_coordinates[i + 1], road_coordinates[i + 2])
        curvatures.append(curvature)
    
    # Pad to match dimensions with lengths (which has one more element)
    while len(angles) < len(lengths):
        angles.append(angles[-1] if angles else 0)
    while len(curvatures) < len(lengths):
        curvatures.append(curvatures[-1] if curvatures else 0)
    
    return angles, lengths, curvatures


def pad_sequences_robust(sequences, max_length=50, padding='post'):
    """Robust sequence padding - FIXED to use correct max_length of 50"""
    if not sequences:
        return np.array([]), 0
        
    # Use exactly 50 as expected by the model
    max_length = 50
    
    padded = []
    for seq in sequences:
        if len(seq) >= max_length:
            # Take the middle part of the sequence for better representation
            start_idx = (len(seq) - max_length) // 2
            padded.append(seq[start_idx:start_idx + max_length])
        else:
            if padding == 'post':
                padded.append(seq + [0] * (max_length - len(seq)))
            else:
                padded.append([0] * (max_length - len(seq)) + seq)
    
    return np.array(padded), max_length


class MyThreeFeatureSelector(competition_pb2_grpc.CompetitionToolServicer):
    """
    Three-feature test selector using trained CNN+BiLSTM model
    Features: turning angles, segment lengths, and curvature
    """
    
    def __init__(self):
        self.model_path = 'best_model.onnx'
        self.scaler_path = 'best_model_scaler.pkl'
        self.onnx_session = None
        self.scaler = None
        self.sequence_length = 50  # Model expects sequences of length 50
        self.is_initialized = False
        
    def _load_model_and_scaler(self):
        """Load the ONNX model and scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                print("ONNX model loaded successfully!")
                self.onnx_session = ort.InferenceSession(self.model_path)
                
                # Print model input shape for debugging
                input_shape = self.onnx_session.get_inputs()[0].shape
                print(f"Expected input shape: {input_shape}")
                
                self.scaler = joblib.load(self.scaler_path)
                print("✓ Scaler loaded successfully!")
                self.is_initialized = True
            else:
                print(f"Model files not found: {self.model_path} or {self.scaler_path}")
                self.is_initialized = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_initialized = False

    def Name(self, request, context):
        return competition_pb2.NameReply(name="MyThreeFeatureSelector")

    def Initialize(self, request_iterator, context):
        """Initialize the test selector with oracle data"""
        print("start test selector")
        
        # Load model and scaler
        self._load_model_and_scaler()
        
        if not self.is_initialized:
            print("Failed to initialize model - will use fallback selection")
        
        print(f"Using sequence length: {self.sequence_length}")
        
        # Process oracle data (training examples)
        oracle_count = 0
        for oracle in request_iterator:
            oracle: competition_pb2.Oracle = oracle
            oracle_count += 1
            if oracle_count <= 5:  # Print first few for debugging
                print("hasFailed={}\ttestId={}".format(oracle.hasFailed, oracle.testCase.testId))
        
        print(f"Processed {oracle_count} oracle examples")
        return competition_pb2.InitializationReply(ok=True)

    def _extract_road_features(self, sdc_test_case):
        """Extract road features from test case - FIXED for correct protobuf structure"""
        try:
            # Extract road points from the test case
            road_points = []
            
            # Based on your protobuf: message SDCTestCase has "repeated RoadPoint roadPoints = 2"
            if hasattr(sdc_test_case, 'roadPoints') and sdc_test_case.roadPoints:
                # Sort by sequenceNumber to ensure correct order
                sorted_points = sorted(sdc_test_case.roadPoints, key=lambda p: p.sequenceNumber)
                
                # Extract x, y coordinates from each RoadPoint
                for point in sorted_points:
                    if hasattr(point, 'x') and hasattr(point, 'y'):
                        road_points.append([float(point.x), float(point.y)])
                    else:
                        print(f"Warning: RoadPoint missing x or y field in test {sdc_test_case.testId}")
            else:
                print(f"No roadPoints found for test {sdc_test_case.testId}")
                return None, None, None
            
            if len(road_points) < 3:
                print(f"Not enough road points for test {sdc_test_case.testId}: {len(road_points)} points")
                return None, None, None
                
            # Extract three features
            angles, lengths, curvatures = extract_three_features(road_points)
            
            if not angles or not lengths or not curvatures:
                print(f"Could not extract features for test {sdc_test_case.testId}")
                return None, None, None
                
            return angles, lengths, curvatures
            
        except Exception as e:
            print(f"Error extracting features for test {sdc_test_case.testId}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def _predict_failure_probability(self, angles, lengths, curvatures):
        """Predict failure probability using the trained model"""
        if not self.is_initialized or not angles or not lengths or not curvatures:
            return 0.1  # Default low probability if model not loaded or no features
        
        try:
            # Pad sequences to model's expected length (50)
            X_angles_padded, _ = pad_sequences_robust([angles], max_length=self.sequence_length)
            X_lengths_padded, _ = pad_sequences_robust([lengths], max_length=self.sequence_length)
            X_curvatures_padded, _ = pad_sequences_robust([curvatures], max_length=self.sequence_length)
            
            # Stack features (shape: [1, sequence_length, 3])
            X = np.stack([X_angles_padded, X_lengths_padded, X_curvatures_padded], axis=-1)
            
            # Ensure correct shape: [1, 50, 3]
            if X.shape != (1, 50, 3):
                print(f"Warning: Expected shape (1, 50, 3), got {X.shape}")
                return 0.1
            
            # Scale features
            X_reshaped = X.reshape(-1, X.shape[-1])  # Shape: [50, 3]
            X_scaled = self.scaler.transform(X_reshaped).reshape(X.shape)  # Back to [1, 50, 3]
            
            # Make prediction
            input_name = self.onnx_session.get_inputs()[0].name
            prediction = self.onnx_session.run(None, {input_name: X_scaled.astype(np.float32)})[0]
            
            prob = float(prediction[0][0])  # Extract probability as float
            
            # Ensure probability is in valid range
            prob = max(0.0, min(1.0, prob))
            
            return prob
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            return 0.1  # Return default probability on error

    def Select(self, request_iterator, context):
        """Select tests based on failure probability"""
        print("Starting test selection process...")
        
        test_probabilities = []
        test_cases = []
        successful_predictions = 0
        failed_predictions = 0
        
        # Process all test cases and calculate probabilities
        for sdc_test_case in request_iterator:
            sdc_test_case: competition_pb2.SDCTestCase = sdc_test_case
            test_cases.append(sdc_test_case)
            
            # Extract features
            angles, lengths, curvatures = self._extract_road_features(sdc_test_case)
            
            if angles is not None and lengths is not None and curvatures is not None:
                # Predict failure probability
                probability = self._predict_failure_probability(angles, lengths, curvatures)
                successful_predictions += 1
            else:
                print(f"Error extracting features for test {sdc_test_case.testId}: insufficient road points or feature extraction failed")
                probability = 0.1  # Default probability for failed feature extraction
                failed_predictions += 1
                
            test_probabilities.append(probability)
            print(f"Test {sdc_test_case.testId}: probability = {probability:.4f}")
        
        if not test_probabilities:
            print("No test probabilities calculated")
            return
        
        # Print extraction success rate
        total_tests = len(test_probabilities)
        print(f"\nFEATURE EXTRACTION RESULTS:")
        print(f"   Successful predictions: {successful_predictions}")
        print(f"   Failed predictions: {failed_predictions}")
        print(f"   Success rate: {100*successful_predictions/total_tests:.1f}%")
        
        # Analysis and selection
        test_probabilities = np.array(test_probabilities)
        
        print(f"\nPREDICTION ANALYSIS:")
        print(f"   Total tests processed: {total_tests}")
        print(f"   Probability statistics:")
        print(f"      Min: {np.min(test_probabilities):.4f}")
        print(f"      Max: {np.max(test_probabilities):.4f}")
        print(f"      Mean: {np.mean(test_probabilities):.4f}")
        print(f"      Median: {np.median(test_probabilities):.4f}")
        print(f"      75th percentile: {np.percentile(test_probabilities, 75):.4f}")
        
        # Target: select 25% of tests (as per your evaluation showing 48/192)
        target_selection_count = max(1, int(total_tests * 0.25))
        
        # Selection strategy: prioritize high-probability tests
        sorted_indices = np.argsort(test_probabilities)[::-1]  # Sort descending
        
        # Tier 1: High probability tests (top 30% threshold)
        high_prob_threshold = np.percentile(test_probabilities, 70)  # Top 30%
        tier1_indices = [i for i in sorted_indices if test_probabilities[i] >= high_prob_threshold]
        
        # If all probabilities are the same (like 0.1000), use random selection from top indices
        if np.std(test_probabilities) < 1e-6:  # All probabilities are nearly identical
            print("   All probabilities are identical - using random selection")
            selected_indices = sorted_indices[:target_selection_count]
        else:
            # Normal selection with tiers
            tier1_count = len(tier1_indices)
            remaining_needed = target_selection_count - tier1_count
            
            tier2_indices = []
            if remaining_needed > 0:
                remaining_indices = [i for i in sorted_indices if i not in tier1_indices]
                tier2_indices = remaining_indices[:remaining_needed]
            
            selected_indices = tier1_indices + tier2_indices
            selected_indices = selected_indices[:target_selection_count]  # Ensure we don't exceed target
        
        print(f"   Target selection: {target_selection_count} tests ({100*target_selection_count/total_tests:.1f}%)")
        print(f"   Tier 1 (Top 30%, prob >={high_prob_threshold:.3f}): {len(tier1_indices)} tests")
        
        print(f"\nFINAL SELECTION RESULTS:")
        print(f"   Total selected: {len(selected_indices)}")
        print(f"   Selection rate: {100*len(selected_indices)/total_tests:.1f}%")
        if selected_indices:
            selected_probs = test_probabilities[selected_indices]
            print(f"   Selected probability range: {np.min(selected_probs):.3f} - {np.max(selected_probs):.3f}")
            print(f"   Selected mean probability: {np.mean(selected_probs):.3f}")
        
        # Show top 10 selected tests
        if selected_indices:
            print(f"\nTOP 10 SELECTED TESTS:")
            top_10 = selected_indices[:min(10, len(selected_indices))]
            for i, idx in enumerate(top_10):
                test_id = test_cases[idx].testId
                prob = test_probabilities[idx]
                print(f"   {i+1:2d}. Test {test_id}: {prob:.4f} ({prob*100:.1f}%)")
        
        # Yield selected tests
        for idx in selected_indices:
            yield competition_pb2.SelectionReply(testId=test_cases[idx].testId)


if __name__ == "__main__":
    print("start test selector")
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port")
    args = parser.parse_args()
    
    GRPC_PORT = args.port
    GRPC_URL = "[::]:" + GRPC_PORT
    
    server = grpc.server(fut.ThreadPoolExecutor(max_workers=2))
    competition_pb2_grpc.add_CompetitionToolServicer_to_server(MyThreeFeatureSelector(), server)
    server.add_insecure_port(GRPC_URL)
    
    print("start server on port {}".format(GRPC_PORT))
    server.start()
    print("server is running")
    server.wait_for_termination()
    print("server terminated")