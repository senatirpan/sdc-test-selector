import random
import argparse
import os
import competition_pb2_grpc
import competition_pb2
import grpc
import concurrent.futures as fut
import numpy as np
import math
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class MyTestSelector(competition_pb2_grpc.CompetitionToolServicer):
    """
    ML-based test selector implementing multiple models:
    1. Logistic Regression
    2. Random Forest 
    3. SVM 
    4. Ensemble voting
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = []
        
    def Name(self, request, context):
        return competition_pb2.NameReply(name="my_test_selector")
    
    def extract_ml_features(self, road_points):
        """
        Extract comprehensive features for ML models
        """
        if len(road_points) < 3:
            return np.zeros(15)  # Return zero features for invalid roads
            
        points = [(p.x, p.y) for p in road_points]
        features = []
        
        # 1. BASIC GEOMETRIC FEATURES
        # Total distance
        distances = [math.sqrt((points[i+1][0] - points[i][0])**2 + 
                              (points[i+1][1] - points[i][1])**2) 
                    for i in range(len(points)-1)]
        total_distance = sum(distances)
        features.extend([
            total_distance,
            len(points),  # Number of road points
            np.mean(distances) if distances else 0,  # Average segment length
            np.std(distances) if distances else 0,   # Distance variation
        ])
        
        # 2. CURVATURE FEATURES (Most important according to research)
        angles = []
        curvatures = []
        
        for i in range(1, len(points)-1):
            # Vectors between consecutive points
            v1 = np.array([points[i][0] - points[i-1][0], points[i][1] - points[i-1][1]])
            v2 = np.array([points[i+1][0] - points[i][0], points[i+1][1] - points[i][1]])
            
            # Calculate angle between vectors
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = math.acos(cos_angle)
                angles.append(angle)
                
                # Curvature (research shows this is critical)
                curvature = angle / (np.mean(distances) if distances else 1)
                curvatures.append(curvature)
        
        if angles and curvatures:
            features.extend([
                max(curvatures),        # Maximum curvature
                np.mean(curvatures),    # Average curvature  
                np.std(curvatures),     # Curvature variation
                max(angles),            # Sharpest turn
                np.mean(angles),        # Average turn angle
            ])
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # 3. COMPLEXITY FEATURES
        sharp_turns = sum(1 for a in angles if a > math.pi/3) if angles else 0  # >60 degrees
        very_sharp_turns = sum(1 for a in angles if a > math.pi/2.5) if angles else 0  # >72 degrees
        
        features.extend([
            sharp_turns,
            very_sharp_turns,
            sharp_turns / len(points) if len(points) > 0 else 0,  # Turn density
        ])
        
        # 4. SEQUENCE FEATURES (for LSTM-like analysis)
        # Direction changes
        direction_changes = 0
        for i in range(2, len(points)):
            v_prev = np.array([points[i-1][0] - points[i-2][0], points[i-1][1] - points[i-2][1]])
            v_curr = np.array([points[i][0] - points[i-1][0], points[i][1] - points[i-1][1]])
            
            if np.linalg.norm(v_prev) > 0 and np.linalg.norm(v_curr) > 0:
                cos_change = np.dot(v_prev, v_curr) / (np.linalg.norm(v_prev) * np.linalg.norm(v_curr))
                if cos_change < 0.5:  # Significant direction change
                    direction_changes += 1
        
        features.extend([
            direction_changes,
            direction_changes / len(points) if len(points) > 0 else 0,  # Direction change rate
        ])
        
        # 5. STATISTICAL FEATURES
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        features.extend([
            np.std(x_coords),  # Spread in x direction
            np.std(y_coords),  # Spread in y direction  
        ])
        
        return np.array(features)
    
    def initialize_models(self):
        """
        Initialize different ML models based on research findings
        """
        # 1. Logistic Regression
        self.models['logistic'] = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        # 2. Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # 3. SVM
        self.models['svm'] = SVC(
            kernel='rbf',
            probability=True,  # Enable probability predictions
            random_state=42,
            class_weight='balanced'
        )
        
        # Feature names for interpretation
        self.feature_names = [
            'total_distance', 'num_points', 'avg_segment', 'distance_std',
            'max_curvature', 'avg_curvature', 'curvature_std', 'max_angle', 'avg_angle',
            'sharp_turns', 'very_sharp_turns', 'turn_density',
            'direction_changes', 'direction_change_rate',
            'x_spread', 'y_spread'
        ]
    
    def train_and_evaluate_models(self, X, y):
        """
        Train all models and select the best one using cross-validation
        """
        print("Training and evaluating ML models...")
        
        best_score = 0
        best_model_name = None
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1')
            mean_score = np.mean(cv_scores)
            
            print(f"{name} CV F1-Score: {mean_score:.4f} (±{np.std(cv_scores):.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_model_name = name
        
        # Train the best model on full data
        print(f"\nBest model: {best_model_name} (F1-Score: {best_score:.4f})")
        self.best_model = self.models[best_model_name]
        self.best_model.fit(X, y)
        
        return best_model_name, best_score
    
    def Initialize(self, request_iterator, context):
        """
        Train ML models using historical test outcomes
        """
        print("Collecting training data from historical tests...")
        
        X_train = []
        y_train = []
        total_tests = 0
        failures = 0
        
        # Initialize models
        self.initialize_models()
        
        for oracle in request_iterator:
            oracle: competition_pb2.Oracle = oracle
            test_id = oracle.testCase.testId
            has_failed = oracle.hasFailed
            
            # Extract features
            features = self.extract_ml_features(oracle.testCase.roadPoints)
            X_train.append(features)
            y_train.append(1 if has_failed else 0)  # 1=FAIL, 0=PASS
            
            if has_failed:
                failures += 1
            total_tests += 1
            
            if total_tests % 1000 == 0:
                print(f"Processed {total_tests} training samples...")
        
        print(f"Training complete: {total_tests} tests, {failures} failures ({failures/total_tests:.1%})")
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train and evaluate models
        best_model_name, best_score = self.train_and_evaluate_models(X_train_scaled, y_train)
        
        # Final evaluation on training data (for debugging)
        y_pred = self.best_model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_pred)
        print(f"\nFinal training accuracy: {train_accuracy:.4f}")
        print(f"Expected test selection strategy: Target ~40-60% selection with high precision")
        
        return competition_pb2.InitializationReply(ok=True)
    
    def Select(self, request_iterator, context):
        """
        Use trained ML model to select tests
        """
        print("Using ML model for test selection...")
        
        if self.best_model is None:
            print("WARNING: No trained model available, falling back to random selection")
            # Fallback to random selection
            for sdc_test_case in request_iterator:
                if random.random() < 0.3:  # Select 30% randomly
                    yield competition_pb2.SelectionReply(testId=sdc_test_case.testId)
            return
        
        test_predictions = []
        
        # Collect and predict all tests
        for sdc_test_case in request_iterator:
            test_id = sdc_test_case.testId
            features = self.extract_ml_features(sdc_test_case.roadPoints)
            
            # Scale features and predict
            features_scaled = self.scaler.transform([features])
            failure_prob = self.best_model.predict_proba(features_scaled)[0][1]  # Probability of failure
            
            test_predictions.append((test_id, failure_prob))
        
        print(f"Analyzed {len(test_predictions)} test cases")
        
        # Selection strategy based on ML predictions
        test_predictions.sort(key=lambda x: x[1], reverse=True)  # Sort by failure probability
        
        # Adaptive selection based on probability distribution
        high_prob_threshold = 0.7   # Very likely to fail
        medium_prob_threshold = 0.4  # Moderately likely to fail
        
        selected_tests = []
        
        # Select tests based on probability tiers
        for test_id, prob in test_predictions:
            if prob >= high_prob_threshold:
                selected_tests.append(test_id)  # Select ALL high probability failures
            elif prob >= medium_prob_threshold:
                if random.random() < 0.6:  # Select 60% of medium probability
                    selected_tests.append(test_id)
            else:
                if random.random() < 0.15:  # Select 15% of low probability (for diversity)
                    selected_tests.append(test_id)
        
        print(f"Selected {len(selected_tests)} tests based on ML predictions")
        print(f"Selection rate: {len(selected_tests)/len(test_predictions):.1%}")
        
        # Yield selected tests
        for test_id in selected_tests:
            yield competition_pb2.SelectionReply(testId=test_id)


if __name__ == "__main__":
    print("Starting ML-Based Test Selector")
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
