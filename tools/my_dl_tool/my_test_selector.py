import argparse
import grpc
import concurrent.futures as fut
import onnxruntime
import numpy as np
import math
import competition_pb2
import competition_pb2_grpc
from scipy.interpolate import interp1d

class RobustScaler:
    def __init__(self):
        self.center_ = None
        self.scale_ = None
    
    def fit_transform(self, X):
        # Calculate median and IQR for robust scaling
        X = np.array(X)
        self.center_ = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        self.scale_ = q75 - q25
        self.scale_[self.scale_ == 0] = 1.0  # Avoid division by zero
        return (X - self.center_) / self.scale_
    
    def transform(self, X):
        X = np.array(X)
        return (X - self.center_) / self.scale_

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points with improved handling"""
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    if np.allclose(v1, 0) or np.allclose(v2, 0):
        return 0
    
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    if norms == 0:
        return 0
    
    cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    angle_deg = np.degrees(angle)
    return 180 - angle_deg

def calculate_curvature(p1, p2, p3):
    """Calculate curvature at point p2"""
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    if np.allclose(v1, 0) or np.allclose(v2, 0):
        return 0
    
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    v_norm = np.linalg.norm(v1 + v2)
    
    if v_norm == 0:
        return 0
    
    return abs(cross) / (v_norm ** 3)

def calculate_segment_length(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def extract_enhanced_features(road_coordinates):
    """Extract enhanced features including curvature and velocity changes"""
    if len(road_coordinates) < 3:
        return [], [], [], []
    
    angles = []
    lengths = []
    curvatures = []
    velocity_changes = []
    
    # Calculate segment lengths
    for i in range(len(road_coordinates) - 1):
        length = calculate_segment_length(road_coordinates[i], road_coordinates[i + 1])
        lengths.append(length)
    
    # Calculate angles and curvatures
    for i in range(len(road_coordinates) - 2):
        angle = calculate_angle(road_coordinates[i], road_coordinates[i + 1], road_coordinates[i + 2])
        angles.append(angle)
        
        curvature = calculate_curvature(road_coordinates[i], road_coordinates[i + 1], road_coordinates[i + 2])
        curvatures.append(curvature)
    
    # Calculate velocity changes
    for i in range(len(lengths) - 1):
        if lengths[i] == 0:
            velocity_changes.append(0)
        else:
            velocity_change = abs(lengths[i+1] - lengths[i]) / lengths[i]
            velocity_changes.append(velocity_change)
    
    # Pad to match dimensions
    target_len = len(lengths)
    
    while len(angles) < target_len:
        angles.append(angles[-1] if angles else 0)
    while len(curvatures) < target_len:
        curvatures.append(curvatures[-1] if curvatures else 0)
    while len(velocity_changes) < target_len:
        velocity_changes.append(velocity_changes[-1] if velocity_changes else 0)
    
    angles = angles[:target_len]
    curvatures = curvatures[:target_len]
    velocity_changes = velocity_changes[:target_len]
    
    return angles, lengths, curvatures, velocity_changes

def adjust_features_to_target_size(features_list, target_size=50):
    """Adjust feature arrays to target size using interpolation or truncation"""
    adjusted_features = []
    
    for features in features_list:
        features = np.array(features)
        
        if len(features) == 0:
            adjusted_features.append(np.zeros(target_size))
            continue
            
        if len(features) == target_size:
            adjusted_features.append(features)
        elif len(features) > target_size:
            indices = np.linspace(0, len(features) - 1, target_size, dtype=int)
            adjusted_features.append(features[indices])
        else:
            if len(features) == 1:
                adjusted_features.append(np.full(target_size, features[0]))
            else:
                current_indices = np.arange(len(features))
                target_indices = np.linspace(0, len(features) - 1, target_size)
                interpolator = interp1d(current_indices, features, kind='linear', 
                                      fill_value='extrapolate', bounds_error=False)
                interpolated = interpolator(target_indices)
                interpolated = np.nan_to_num(interpolated, nan=0.0)
                adjusted_features.append(interpolated)
    
    return adjusted_features

# Feature statistics from training data
FEATURE_STATS = {
    'center_': np.array([174.532292, 3.455451, 0.00347680, 0.002703]),
    'scale_': np.array([5.732392, 1.023584, 0.00360055, 0.004398])
}

class MyTestSelector(competition_pb2_grpc.CompetitionToolServicer):
    def __init__(self):
        self.onnx_session = None
        
    def Name(self, request, context):
        return competition_pb2.NameReply(name="MyEnhancedSelector")

    def Initialize(self, request_iterator, context):
        # Load ONNX model
        try:
            self.onnx_session = onnxruntime.InferenceSession("best_model_fold_3.onnx")
            print(f"ONNX model loaded successfully!")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            
        for oracle in request_iterator:
            oracle: competition_pb2.Oracle = oracle
        return competition_pb2.InitializationReply(ok=True)

    def Select(self, request_iterator, context):
        print(f"Starting test selection process...")
        
        if not self.onnx_session:
            print("No ONNX model loaded!")
            return
        
        all_predictions = []
        
        # First pass: collect all predictions
        for sdc_test_case in request_iterator:
            road_coordinates = [[point.x, point.y] for point in sdc_test_case.roadPoints]
            
            try:
                # Extract and process features
                angles, lengths, curvatures, velocity_changes = extract_enhanced_features(road_coordinates)
                
                if len(lengths) == 0:
                    # Assign low probability for invalid cases
                    all_predictions.append((sdc_test_case.testId, 0.1, road_coordinates))
                    continue
                
                # Adjust to target size and create input
                adjusted_features = adjust_features_to_target_size(
                    [angles, lengths, curvatures, velocity_changes], 50
                )
                
                feature_input_data = np.stack(adjusted_features, axis=-1).astype(np.float32)
                feature_input_data = feature_input_data.reshape(1, 50, 4)
                
                # Apply scaling
                original_shape = feature_input_data.shape
                feature_flat = feature_input_data.reshape(-1, 4)
                scaled_features = (feature_flat - FEATURE_STATS['center_']) / FEATURE_STATS['scale_']
                scaled_features = np.nan_to_num(scaled_features, nan=0.0).astype(np.float32)
                feature_input_data_scaled = scaled_features.reshape(original_shape)
                
                # Run prediction
                prediction = self.onnx_session.run(
                    None, 
                    {self.onnx_session.get_inputs()[0].name: feature_input_data_scaled}
                )
                
                fail_probability = float(prediction[0][0][0])
                all_predictions.append((sdc_test_case.testId, fail_probability, road_coordinates))
                
            except Exception as e:
                print(f"Error processing test {sdc_test_case.testId}: {e}")
                all_predictions.append((sdc_test_case.testId, 0.1, road_coordinates))
                continue
        
        # Enhanced selection strategy
        if not all_predictions:
            return
            
        # Sort by probability (descending)
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        
        probabilities = [p[1] for p in all_predictions]
        total_tests = len(all_predictions)
        
        print(f"\nPREDICTION ANALYSIS:")
        print(f"   Total tests processed: {total_tests}")
        print(f"   Probability statistics:")
        print(f"      Min: {min(probabilities):.4f}")
        print(f"      Max: {max(probabilities):.4f}")
        print(f"      Mean: {np.mean(probabilities):.4f}")
        print(f"      Median: {np.median(probabilities):.4f}")
        print(f"      75th percentile: {np.percentile(probabilities, 75):.4f}")
        
        # Fixed-target selection strategy - ensure we always select the right amount
        target_selection_count = max(int(total_tests * 0.25), 48)  # Target 25% or minimum 48 tests
        max_selection_count = int(total_tests * 0.35)  # Cap at 35%
        
        print(f"   Target selection: {target_selection_count} tests ({target_selection_count/total_tests:.1%})")
        
        selected_tests = []
        
        # Strategy 1: Top probability selections (get the best ones first)
        top_30_percent = int(total_tests * 0.30)
        high_prob_candidates = all_predictions[:top_30_percent]  # Already sorted by probability
        
        # Select all from top 30% with decent probability
        high_threshold = max(0.2, np.percentile(probabilities, 70))
        
        for test_id, prob, coords in high_prob_candidates:
            if prob >= high_threshold:
                selected_tests.append(test_id)
        
        tier1_count = len(selected_tests)
        print(f"   Tier 1 (Top 30%, prob >={high_threshold:.3f}): {tier1_count} tests")
        
        # Strategy 2: Fill up to target with next best + diversity
        remaining_needed = target_selection_count - len(selected_tests)
        
        if remaining_needed > 0:
            print(f"   Need {remaining_needed} more tests to reach target")
            
            # Get remaining candidates (not yet selected)
            remaining_candidates = [(test_id, prob, coords) for test_id, prob, coords in all_predictions 
                                  if test_id not in selected_tests]
            
            # Strategy 2a: Take next highest probabilities
            next_best_count = min(remaining_needed // 2, len(remaining_candidates))
            for i in range(next_best_count):
                test_id, prob, coords = remaining_candidates[i]
                selected_tests.append(test_id)
                remaining_needed -= 1
            
            # Strategy 2b: Add diversity-based selections
            if remaining_needed > 0:
                # Calculate complexity scores for remaining candidates
                complexity_candidates = []
                for test_id, prob, coords in remaining_candidates[next_best_count:]:
                    complexity = self.calculate_road_complexity(coords)
                    complexity_candidates.append((test_id, prob, coords, complexity))
                
                # Sort by complexity (descending) for diversity
                complexity_candidates.sort(key=lambda x: x[3], reverse=True)
                
                # Take most complex remaining tests
                diversity_count = min(remaining_needed, len(complexity_candidates))
                for i in range(diversity_count):
                    test_id, prob, coords, complexity = complexity_candidates[i]
                    selected_tests.append(test_id)
                    remaining_needed -= 1
            
            tier2_count = len(selected_tests) - tier1_count
            print(f"   Tier 2 (Fill to target): {tier2_count} tests")
        
        # Strategy 3: Ensure we have minimum selection if still short
        if len(selected_tests) < int(total_tests * 0.20):  # Absolute minimum 20%
            minimum_needed = int(total_tests * 0.20) - len(selected_tests)
            print(f"   Still need {minimum_needed} tests for minimum 20% coverage")
            
            remaining_unselected = [test_id for test_id, prob, coords in all_predictions 
                                  if test_id not in selected_tests]
            
            # Just take the next best available
            for i in range(min(minimum_needed, len(remaining_unselected))):
                selected_tests.append(remaining_unselected[i])
            
            tier3_count = len(selected_tests) - tier1_count - (tier2_count if 'tier2_count' in locals() else 0)
            print(f"   Tier 3 (Minimum guarantee): {tier3_count} tests")
        
        # Final statistics
        selection_rate = len(selected_tests) / total_tests
        selected_probs = [prob for test_id, prob, coords in all_predictions if test_id in selected_tests]
        
        print(f"\nFINAL SELECTION RESULTS:")
        print(f"   Total selected: {len(selected_tests)}")
        print(f"   Selection rate: {selection_rate:.1%}")
        print(f"   Selected probability range: {min(selected_probs):.3f} - {max(selected_probs):.3f}")
        print(f"   Selected mean probability: {np.mean(selected_probs):.3f}")
        
        # Show top selections
        print(f"\nTOP 10 SELECTED TESTS:")
        selected_with_probs = [(test_id, prob) for test_id, prob, coords in all_predictions if test_id in selected_tests]
        selected_with_probs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (test_id, prob) in enumerate(selected_with_probs[:10]):
            print(f"   {i+1:2d}. Test {test_id}: {prob:.4f} ({prob*100:.1f}%)")
        
        # Return selections
        for test_id in selected_tests:
            yield competition_pb2.SelectionReply(testId=test_id)
    
    def calculate_road_complexity(self, coords):
        """Calculate a simple complexity score for the road"""
        if len(coords) < 3:
            return 0.0
        
        # Calculate total path length and total angular change
        total_length = 0
        total_angle_change = 0
        
        for i in range(len(coords) - 1):
            segment_length = calculate_segment_length(coords[i], coords[i + 1])
            total_length += segment_length
        
        for i in range(len(coords) - 2):
            angle = calculate_angle(coords[i], coords[i + 1], coords[i + 2])
            if angle > 10:  # Only count significant turns
                total_angle_change += angle
        
        # Normalize complexity score (0-1 range)
        length_factor = min(total_length / 100.0, 1.0)  # Normalize by 100 units
        angle_factor = min(total_angle_change / 1000.0, 1.0)  # Normalize by 1000 degrees
        
        return (length_factor + angle_factor) / 2.0

if __name__ == "__main__":
    print("start test selector")
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default="50051")
    args = parser.parse_args()
    GRPC_PORT = args.port
    
    server = grpc.server(fut.ThreadPoolExecutor(max_workers=4))
    competition_pb2_grpc.add_CompetitionToolServicer_to_server(MyTestSelector(), server)
    
    server.add_insecure_port("[::]:" + args.port)
    print("start server on port {}".format(GRPC_PORT))
    server.start()
    print("server is running")
    server.wait_for_termination()
    print("server terminated")