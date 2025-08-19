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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import pickle

class MyTestSelector(competition_pb2_grpc.CompetitionToolServicer):
    """
    Intelligent test selector that optimizes for both fault detection and diversity
    using road geometry analysis, failure prediction, and clustering-based selection.
    """
    
    def __init__(self):
        self.historical_data = {}  # testId -> failure probability
        self.feature_scaler = StandardScaler()
        self.failure_model = None
        self.diversity_clusters = None
        self.test_features = {}  # Cache for computed features
        self.initialization_complete = False
        
    def Name(self, request, context):
        return competition_pb2.NameReply(name="my_test_selector")
    
    def extract_features(self, road_points):
        """Extract meaningful features from road points that correlate with failures"""
        if not road_points:
            return np.zeros(8)
            
        points = [(p.x, p.y) for p in road_points]
        
        # Distance and geometric features
        distances = [math.sqrt((points[i+1][0] - points[i][0])**2 + 
                              (points[i+1][1] - points[i][1])**2) 
                    for i in range(len(points)-1)]
        total_distance = sum(distances)
        avg_segment_length = np.mean(distances) if distances else 0
        
        # Curvature analysis
        angles = []
        for i in range(1, len(points)-1):
            v1 = np.array([points[i][0] - points[i-1][0], points[i][1] - points[i-1][1]])
            v2 = np.array([points[i+1][0] - points[i][0], points[i+1][1] - points[i][1]])
            
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = math.acos(cos_angle)
                angles.append(angle)
        
        max_curvature = max(angles) if angles else 0
        avg_curvature = np.mean(angles) if angles else 0
        curvature_std = np.std(angles) if angles else 0
        sharp_turns = sum(1 for a in angles if a > math.pi/3)  # > 60 degrees
        
        # Road complexity metrics
        direction_changes = 0
        for i in range(2, len(points)):
            # Check if direction changes significantly
            v_prev = np.array([points[i-1][0] - points[i-2][0], points[i-1][1] - points[i-2][1]])
            v_curr = np.array([points[i][0] - points[i-1][0], points[i][1] - points[i-1][1]])
            
            if np.linalg.norm(v_prev) > 0 and np.linalg.norm(v_curr) > 0:
                cos_change = np.dot(v_prev, v_curr) / (np.linalg.norm(v_prev) * np.linalg.norm(v_curr))
                if cos_change < 0.7:  # Significant direction change
                    direction_changes += 1
        
        return np.array([
            total_distance,
            len(points),
            max_curvature,
            avg_curvature,
            curvature_std,
            sharp_turns,
            direction_changes,
            avg_segment_length
        ])
    
    def compute_failure_probability(self, features):
        """Compute failure probability based on features"""
        if self.failure_model is None:
            # Simple heuristic model based on road complexity
            complexity_score = (
                features[2] * 0.3 +      # max_curvature
                features[4] * 0.2 +      # curvature_std  
                features[5] * 0.25 +     # sharp_turns
                features[6] * 0.15 +     # direction_changes
                (features[0] / 1000) * 0.1  # normalized distance
            )
            return min(1.0, complexity_score)
        else:
            # Use trained model if available
            return self.failure_model.predict_proba([features])[0][1]
    
    def build_failure_model(self):
        """Build a simple failure prediction model from historical data"""
        failure_rates = defaultdict(list)
        
        for test_id, has_failed in self.historical_data.items():
            if test_id in self.test_features:
                features = self.test_features[test_id]
                # Group by feature similarity and compute failure rates
                complexity = features[2] + features[4] + features[5]  # Combined complexity metric
                failure_rates[round(complexity, 1)].append(1 if has_failed else 0)
        
        # Simple lookup table for failure prediction
        self.failure_lookup = {}
        for complexity, failures in failure_rates.items():
            self.failure_lookup[complexity] = sum(failures) / len(failures)
    
    def Initialize(self, request_iterator, context):
        """Process historical test outcomes to build failure prediction model"""
        print("Starting initialization...")
        
        failure_count = 0
        total_count = 0
        
        for oracle in request_iterator:
            oracle: competition_pb2.Oracle = oracle
            test_id = oracle.testCase.testId
            has_failed = oracle.hasFailed
            
            # Store historical failure data
            self.historical_data[test_id] = has_failed
            if has_failed:
                failure_count += 1
            total_count += 1
            
            # Extract and cache features
            features = self.extract_features(oracle.testCase.roadPoints)
            self.test_features[test_id] = features
            
            if total_count % 1000 == 0:
                print(f"Processed {total_count} test cases...")
        
        print(f"Initialization complete: {total_count} tests, {failure_count} failures")
        print(f"Overall failure rate: {failure_count/total_count:.3f}")
        
        # Build failure prediction model
        self.build_failure_model()
        
        # Create diversity clusters
        if self.test_features:
            features_matrix = np.array(list(self.test_features.values()))
            features_matrix = self.feature_scaler.fit_transform(features_matrix)
            
            # Use reasonable number of clusters (aim for ~50-100 clusters for diversity)
            n_clusters = min(100, max(10, len(self.test_features) // 50))
            self.diversity_clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.diversity_clusters.fit(features_matrix)
            print(f"Created {n_clusters} diversity clusters")
        
        self.initialization_complete = True
        return competition_pb2.InitializationReply(ok=True)
    
    def Select(self, request_iterator, context):
        """Select tests based on failure probability and diversity"""
        print("Starting test selection...")
        
        if not self.initialization_complete:
            print("Warning: Initialization not complete, using fallback selection")
        
        test_candidates = []
        cluster_counts = defaultdict(int)
        
        # First pass: collect all candidates with their scores
        for sdc_test_case in request_iterator:
            sdc_test_case: competition_pb2.SDCTestCase = sdc_test_case
            test_id = sdc_test_case.testId
            
            # Extract features for new test case
            if test_id not in self.test_features:
                features = self.extract_features(sdc_test_case.roadPoints)
                self.test_features[test_id] = features
            else:
                features = self.test_features[test_id]
            
            # Compute failure probability
            failure_prob = self.compute_failure_probability(features)
            
            # Determine cluster for diversity
            cluster_id = -1
            if self.diversity_clusters is not None:
                features_scaled = self.feature_scaler.transform([features])
                cluster_id = self.diversity_clusters.predict(features_scaled)[0]
            
            test_candidates.append({
                'test_id': test_id,
                'failure_prob': failure_prob,
                'cluster_id': cluster_id,
                'features': features
            })
        
        print(f"Evaluating {len(test_candidates)} test candidates...")
        
        # Smart selection strategy
        selected_tests = self.smart_selection(test_candidates)
        
        print(f"Selected {len(selected_tests)} tests")
        
        # Yield selected tests
        for test_id in selected_tests:
            yield competition_pb2.SelectionReply(testId=test_id)
    
    def smart_selection(self, candidates):
        """Implement smart selection balancing fault detection and diversity"""
        if not candidates:
            return []
        
        # Sort candidates by failure probability (descending)
        candidates.sort(key=lambda x: x['failure_prob'], reverse=True)
        
        selected = []
        cluster_counts = defaultdict(int)
        max_per_cluster = max(1, len(candidates) // 20)  # Limit per cluster for diversity
        
        # Selection strategy: 
        # 1. Prioritize high failure probability tests
        # 2. Ensure diversity by limiting selections per cluster
        # 3. Include some random exploration
        
        high_prob_threshold = 0.7
        medium_prob_threshold = 0.3
        
        # Phase 1: Select high-probability failures (up to 40% of selection)
        high_prob_limit = len(candidates) // 5
        for candidate in candidates:
            if len(selected) >= high_prob_limit:
                break
                
            if (candidate['failure_prob'] >= high_prob_threshold and 
                cluster_counts[candidate['cluster_id']] < max_per_cluster):
                selected.append(candidate['test_id'])
                cluster_counts[candidate['cluster_id']] += 1
        
        # Phase 2: Diversified selection from medium probability tests
        medium_prob_candidates = [c for c in candidates 
                                if medium_prob_threshold <= c['failure_prob'] < high_prob_threshold]
        
        # Shuffle for diversity within probability range
        random.shuffle(medium_prob_candidates)
        
        target_selection = len(candidates) // 3  # Target ~33% selection rate
        
        for candidate in medium_prob_candidates:
            if len(selected) >= target_selection:
                break
                
            if cluster_counts[candidate['cluster_id']] < max_per_cluster:
                selected.append(candidate['test_id'])
                cluster_counts[candidate['cluster_id']] += 1
        
        # Phase 3: Fill remaining slots with diverse low-probability tests for exploration
        remaining_candidates = [c for c in candidates if c['test_id'] not in selected]
        random.shuffle(remaining_candidates)
        
        for candidate in remaining_candidates:
            if len(selected) >= target_selection:
                break
                
            if cluster_counts[candidate['cluster_id']] < max_per_cluster // 2:
                selected.append(candidate['test_id'])
                cluster_counts[candidate['cluster_id']] += 1
        
        return selected


if __name__ == "__main__":
    print("Starting Geometric Test Selector")
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", default="4545")
    args = parser.parse_args()
    
    GRPC_PORT = args.port
    GRPC_URL = "[::]:" + GRPC_PORT
    
    server = grpc.server(fut.ThreadPoolExecutor(max_workers=4))
    competition_pb2_grpc.add_CompetitionToolServicer_to_server(MyTestSelector(), server)
    server.add_insecure_port(GRPC_URL)
    
    print(f"Starting server on port {GRPC_PORT}")
    server.start()
    print("Server is running...")
    server.wait_for_termination()
    print("Server terminated")