import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, BatchNormalization,
                                   Conv1D, MaxPooling1D, Bidirectional, GlobalMaxPooling1D, 
                                   Concatenate)
from tensorflow.keras.regularizers import l1_l2
import tf2onnx
import onnx
from sklearn.preprocessing import RobustScaler
import json
import math
import h5py
import pandas as pd

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
    
    # Calculate angles and curvatures (need at least 3 points)
    for i in range(len(road_coordinates) - 2):
        angle = calculate_angle(road_coordinates[i], road_coordinates[i + 1], road_coordinates[i + 2])
        angles.append(angle)
        
        curvature = calculate_curvature(road_coordinates[i], road_coordinates[i + 1], road_coordinates[i + 2])
        curvatures.append(curvature)
    
    # Calculate velocity changes (acceleration proxy)
    for i in range(len(lengths) - 1):
        if lengths[i] == 0:
            velocity_changes.append(0)
        else:
            velocity_change = abs(lengths[i+1] - lengths[i]) / lengths[i]
            velocity_changes.append(velocity_change)
    
    # Pad to match dimensions
    while len(angles) < len(lengths):
        angles.append(angles[-1] if angles else 0)
    while len(curvatures) < len(lengths):
        curvatures.append(curvatures[-1] if curvatures else 0)
    while len(velocity_changes) < len(lengths):
        velocity_changes.append(velocity_changes[-1] if velocity_changes else 0)
    
    return angles, lengths, curvatures, velocity_changes

def load_test_data_for_evaluation(data_dir, max_files=100):
    """Load a small subset of test data for model evaluation"""
    print("Loading test data for model evaluation...")
    
    X_angles = []
    X_lengths = []
    X_curvatures = []
    X_velocity_changes = []
    y = []
    
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')][:max_files]
    
    for file in json_files:
        file_path = os.path.join(data_dir, file)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            road_coordinates = data.get('road_points', [])
            test_outcome = data.get('test_outcome', 'PASS')
            
            if not road_coordinates:
                continue
            
            label_binary = 1 if test_outcome == 'FAIL' else 0

            
            # Clean road coordinates
            clean_coordinates = []
            for point in road_coordinates:
                if isinstance(point, list) and len(point) >= 2:
                    clean_coordinates.append([float(point[0]), float(point[1])])
            
            if len(clean_coordinates) >= 10:
                angles, lengths, curvatures, velocity_changes = extract_enhanced_features(clean_coordinates)
                
                if len(angles) > 0 and len(lengths) > 0:
                    X_angles.append(angles)
                    X_lengths.append(lengths)
                    X_curvatures.append(curvatures)
                    X_velocity_changes.append(velocity_changes)
                    y.append(label_binary)
                    
        except Exception as e:
            continue
    
    return X_angles, X_lengths, X_curvatures, X_velocity_changes, y

def pad_sequences_robust(sequences, max_length=None, padding='post'):
    """Robust sequence padding with better handling"""
    if not sequences:
        return np.array([]), 0
        
    if max_length is None:
        max_length = max(len(seq) for seq in sequences if len(seq) > 0)
    
    max_length = min(max_length, 500)  # Cap to prevent memory issues
    
    padded = []
    for seq in sequences:
        if len(seq) >= max_length:
            start_idx = (len(seq) - max_length) // 2
            padded.append(seq[start_idx:start_idx + max_length])
        else:
            if padding == 'post':
                padded.append(seq + [0] * (max_length - len(seq)))
            else:
                padded.append([0] * (max_length - len(seq)) + seq)
    
    return np.array(padded), max_length

def load_model_weights_only(model_path, input_shape):
    """Load model by reconstructing architecture and loading weights"""
    try:
        # Try to load normally first
        return load_model(model_path)
    except Exception as e:
        print(f"Normal loading failed: {e}")
        print("Attempting to reconstruct model architecture...")
        
        try:
            # Reconstruct the exact architecture from the training code
            # Input layer
            inputs = Input(shape=input_shape, name='input_layer')
            
            # Multi-scale CNN feature extraction (exactly as in training)
            # Scale 1: Fine-grained patterns
            conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
            conv1 = BatchNormalization()(conv1)
            conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(conv1)
            conv1 = BatchNormalization()(conv1)
            pool1 = MaxPooling1D(pool_size=2)(conv1)
            
            # Scale 2: Medium-scale patterns
            conv2 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(inputs)
            conv2 = BatchNormalization()(conv2)
            conv2 = Conv1D(filters=32, kernel_size=5, activation='relu', padding='same')(conv2)
            conv2 = BatchNormalization()(conv2)
            pool2 = MaxPooling1D(pool_size=2)(conv2)
            
            # Scale 3: Large-scale patterns
            conv3 = Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(inputs)
            conv3 = BatchNormalization()(conv3)
            pool3 = MaxPooling1D(pool_size=2)(conv3)
            
            # Concatenate multi-scale features
            concat = Concatenate()([pool1, pool2, pool3])
            concat = Dropout(0.3)(concat)
            
            # Bidirectional LSTM layers
            lstm1 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, 
                                      recurrent_dropout=0.2))(concat)
            lstm1 = BatchNormalization()(lstm1)
            
            lstm2 = Bidirectional(LSTM(32, return_sequences=False, dropout=0.2, 
                                      recurrent_dropout=0.2))(lstm1)
            lstm2 = BatchNormalization()(lstm2)
            
            # Global pooling of CNN features (parallel path)
            global_pool = GlobalMaxPooling1D()(concat)
            
            # Combine LSTM and global pooling features
            combined = Concatenate()([lstm2, global_pool])
            combined = Dropout(0.5)(combined)
            
            # Dense layers with regularization
            dense1 = Dense(64, activation='relu', 
                           kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(combined)
            dense1 = BatchNormalization()(dense1)
            dense1 = Dropout(0.5)(dense1)
            
            dense2 = Dense(32, activation='relu', 
                           kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(dense1)
            dense2 = BatchNormalization()(dense2)
            dense2 = Dropout(0.3)(dense2)
            
            # Output layer
            outputs = Dense(1, activation='sigmoid')(dense2)
            
            model = Model(inputs=inputs, outputs=outputs)
            
            # Load only the weights
            model.load_weights(model_path)
            
            print("✅ Successfully reconstructed model and loaded weights")
            return model
            
        except Exception as e2:
            print(f"❌ Failed to reconstruct model: {e2}")
            
            # Last resort: try to extract weights manually
            try:
                with h5py.File(model_path, 'r') as f:
                    print("Available keys in H5 file:")
                    def print_structure(name, obj):
                        print(name)
                    f.visititems(print_structure)
                    
                print("❌ Manual weight extraction not implemented. Skipping this model.")
                return None
            except Exception as e3:
                print(f"❌ Cannot even read H5 file: {e3}")
                return None

def extract_training_info(model_path):
    """Extract training history and model information from H5 file"""
    training_info = {}
    
    try:
        # Get file size
        file_size = os.path.getsize(model_path)
        training_info['model_size_mb'] = round(file_size / (1024 * 1024), 2)
        
        with h5py.File(model_path, 'r') as f:
            # Try to extract optimizer weights to get training info
            if 'optimizer_weights' in f:
                opt_weights = f['optimizer_weights']
                if 'adamw' in opt_weights:
                    adamw_weights = opt_weights['adamw']
                    if 'iteration' in adamw_weights:
                        iteration = adamw_weights['iteration'][()]
                        training_info['total_epochs'] = int(iteration)
                        training_info['best_epoch'] = int(iteration)
                    
                    if 'learning_rate' in adamw_weights:
                        lr = adamw_weights['learning_rate'][()]
                        training_info['final_learning_rate'] = float(lr)
            
            # Try to get model architecture info
            if 'model_weights' in f:
                model_weights = f['model_weights']
                # Count layers
                layer_count = len(list(model_weights.keys()))
                training_info['total_layers'] = layer_count
                
                # Try to estimate parameters
                total_params = 0
                trainable_params = 0
                for layer_name in model_weights.keys():
                    layer_weights = model_weights[layer_name]
                    if isinstance(layer_weights, h5py.Group):
                        for weight_name in layer_weights.keys():
                            if weight_name in ['kernel', 'bias', 'gamma', 'beta']:
                                weight_data = layer_weights[weight_name]
                                if hasattr(weight_data, 'shape'):
                                    params = np.prod(weight_data.shape)
                                    total_params += params
                                    trainable_params += params
                
                training_info['total_parameters'] = int(total_params)
                training_info['trainable_parameters'] = int(trainable_params)
        
        # Set default values for missing info
        if 'total_epochs' not in training_info:
            training_info['total_epochs'] = 'N/A'
        if 'best_epoch' not in training_info:
            training_info['best_epoch'] = 'N/A'
        if 'final_val_loss' not in training_info:
            training_info['final_val_loss'] = 'N/A'
        if 'best_val_loss' not in training_info:
            training_info['best_val_loss'] = 'N/A'
        if 'final_val_accuracy' not in training_info:
            training_info['final_val_accuracy'] = 'N/A'
        if 'best_val_accuracy' not in training_info:
            training_info['best_val_accuracy'] = 'N/A'
            
    except Exception as e:
        print(f"Warning: Could not extract training info from {model_path}: {e}")
        # Set default values
        training_info = {
            'total_epochs': 'N/A',
            'best_epoch': 'N/A',
            'final_val_loss': 'N/A',
            'best_val_loss': 'N/A',
            'final_val_accuracy': 'N/A',
            'best_val_accuracy': 'N/A',
            'model_size_mb': 'N/A',
            'total_parameters': 'N/A',
            'trainable_parameters': 'N/A'
        }
    
    return training_info

def evaluate_saved_models(model_dir, data_dir, num_folds=10):
    """Evaluate all saved models and find the best one"""
    print(f"Evaluating {num_folds} saved models...")
    
    # Load test data
    X_angles, X_lengths, X_curvatures, X_velocity_changes, y = load_test_data_for_evaluation(data_dir, max_files=200)
    
    if len(X_angles) == 0:
        print("No test data loaded. Please check your data directory.")
        return None, None
    
    # Pad sequences
    X_angles_padded, max_len_angles = pad_sequences_robust(X_angles)
    X_lengths_padded, max_len_lengths = pad_sequences_robust(X_lengths)
    X_curvatures_padded, max_len_curvatures = pad_sequences_robust(X_curvatures)
    X_velocity_changes_padded, max_len_velocity = pad_sequences_robust(X_velocity_changes)
    
    # Ensure all features have the same sequence length
    max_len = max(max_len_angles, max_len_lengths, max_len_curvatures, max_len_velocity)
    
    def pad_to_max_len(arr, current_len, target_len):
        if current_len < target_len:
            return np.pad(arr, ((0, 0), (0, target_len - current_len)), 
                         mode='constant', constant_values=0)
        return arr
    
    X_angles_padded = pad_to_max_len(X_angles_padded, max_len_angles, max_len)
    X_lengths_padded = pad_to_max_len(X_lengths_padded, max_len_lengths, max_len)
    X_curvatures_padded = pad_to_max_len(X_curvatures_padded, max_len_curvatures, max_len)
    X_velocity_changes_padded = pad_to_max_len(X_velocity_changes_padded, max_len_velocity, max_len)
    
    # Stack features
    X_test = np.stack([X_angles_padded, X_lengths_padded, X_curvatures_padded, X_velocity_changes_padded], axis=-1)
    y_test = np.array(y)
    
    # Scale the test data
    scaler = RobustScaler()
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.fit_transform(X_test_reshaped).reshape(X_test.shape)
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels distribution - PASS: {np.sum(y_test == 0)}, FAIL: {np.sum(y_test == 1)}")
    
    # Evaluate each model
    model_performances = []
    input_shape = (X_test.shape[1], X_test.shape[2])  # (sequence_length, features)
    
    for fold in range(1, num_folds + 1):
        model_path = os.path.join(model_dir, f'best_model_fold_{fold}.h5')
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        print(f"\nEvaluating Fold {fold} model...")
        
        try:
            # Load model with fallback to weight-only loading
            model = load_model_weights_only(model_path, input_shape)
            
            if model is None:
                print(f"❌ Could not load model for fold {fold}")
                continue
            
            # Predict
            y_pred_proba = model.predict(X_test_scaled, verbose=0)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
            
            # Extract confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Extract training history information from H5 file
            training_info = extract_training_info(model_path)
            
            performance = {
                'fold': fold,
                'model_path': model_path,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'overall_score': (accuracy + f1 + auc) / 3,  # Combined score
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'confusion_matrix': cm.tolist(),
                'total_epochs': training_info.get('total_epochs', 'N/A'),
                'best_epoch': training_info.get('best_epoch', 'N/A'),
                'final_val_loss': training_info.get('final_val_loss', 'N/A'),
                'best_val_loss': training_info.get('best_val_loss', 'N/A'),
                'final_val_accuracy': training_info.get('final_val_accuracy', 'N/A'),
                'best_val_accuracy': training_info.get('best_val_accuracy', 'N/A'),
                'model_size_mb': training_info.get('model_size_mb', 'N/A'),
                'total_parameters': training_info.get('total_parameters', 'N/A'),
                'trainable_parameters': training_info.get('trainable_parameters', 'N/A')
            }
            
            model_performances.append(performance)
            
            print(f"✅ Fold {fold} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
        except Exception as e:
            print(f"❌ Error evaluating Fold {fold}: {e}")
            continue
    
    if not model_performances:
        print("No models could be evaluated.")
        return None, None
    
    # Find best model based on overall score
    best_model = max(model_performances, key=lambda x: x['overall_score'])
    
    print(f"\n{'='*60}")
    print("MODEL EVALUATION RESULTS")
    print(f"{'='*60}")
    
    for perf in model_performances:
        print(f"Fold {perf['fold']}: Acc={perf['accuracy']:.4f}, F1={perf['f1']:.4f}, "
              f"AUC={perf['auc']:.4f}, Overall={perf['overall_score']:.4f}")
    
    print(f"\nBEST MODEL: Fold {best_model['fold']}")
    print(f"Path: {best_model['model_path']}")
    print(f"Accuracy: {best_model['accuracy']:.4f}")
    print(f"Precision: {best_model['precision']:.4f}")
    print(f"Recall: {best_model['recall']:.4f}")
    print(f"F1-Score: {best_model['f1']:.4f}")
    print(f"AUC: {best_model['auc']:.4f}")
    print(f"Overall Score: {best_model['overall_score']:.4f}")
    
    # Save detailed results to CSV
    save_results_to_csv(model_performances, model_dir)
    
    return best_model, model_performances

def save_results_to_csv(model_performances, model_dir):
    """Save detailed model evaluation results to CSV files"""
    print(f"\nSaving detailed results to CSV files...")
    
    # Create main results CSV
    main_results = []
    for perf in model_performances:
        # Create a clean version for CSV (remove complex objects)
        csv_row = {
            'fold': perf['fold'],
            'model_path': perf['model_path'],
            'accuracy': perf['accuracy'],
            'precision': perf['precision'],
            'recall': perf['recall'],
            'f1': perf['f1'],
            'auc': perf['auc'],
            'overall_score': perf['overall_score'],
            'true_negatives': perf['true_negatives'],
            'false_positives': perf['false_positives'],
            'false_negatives': perf['false_negatives'],
            'true_positives': perf['true_positives'],
            'total_epochs': perf['total_epochs'],
            'best_epoch': perf['best_epoch'],
            'final_val_loss': perf['final_val_loss'],
            'best_val_loss': perf['best_val_loss'],
            'final_val_accuracy': perf['final_val_accuracy'],
            'best_val_accuracy': perf['best_val_accuracy'],
            'model_size_mb': perf['model_size_mb'],
            'total_parameters': perf['total_parameters'],
            'trainable_parameters': perf['trainable_parameters']
        }
        main_results.append(csv_row)
    
    # Save main results
    main_csv_path = os.path.join(model_dir, 'model_evaluation_results.csv')
    df_main = pd.DataFrame(main_results)
    df_main.to_csv(main_csv_path, index=False)
    print(f"✅ Main results saved to: {main_csv_path}")
    
    # Create confusion matrix CSV
    cm_results = []
    for perf in model_performances:
        cm = perf['confusion_matrix']
        cm_results.append({
            'fold': perf['fold'],
            'true_negatives': cm[0][0],
            'false_positives': cm[0][1],
            'false_negatives': cm[1][0],
            'true_positives': cm[1][1],
            'total_samples': cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1],
            'sensitivity': cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0,
            'specificity': cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0,
            'precision': cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0,
            'negative_predictive_value': cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        })
    
    cm_csv_path = os.path.join(model_dir, 'confusion_matrices.csv')
    df_cm = pd.DataFrame(cm_results)
    df_cm.to_csv(cm_csv_path, index=False)
    print(f"✅ Confusion matrices saved to: {cm_csv_path}")
    
    # Save detailed confusion matrix as JSON for better visualization
    cm_detailed = {}
    for perf in model_performances:
        fold = perf['fold']
        cm_detailed[f'fold_{fold}'] = {
            'confusion_matrix': perf['confusion_matrix'],
            'metrics': {
                'true_negatives': int(perf['true_negatives']),
                'false_positives': int(perf['false_positives']),
                'false_negatives': int(perf['false_negatives']),
                'true_positives': int(perf['true_positives']),
                'sensitivity': float(perf['recall']),
                'specificity': float(cm_results[fold-1]['specificity']),
                'precision': float(perf['precision']),
                'accuracy': float(perf['accuracy'])
            }
        }
    
    cm_json_path = os.path.join(model_dir, 'confusion_matrices_detailed.json')
    with open(cm_json_path, 'w') as f:
        json.dump(cm_detailed, f, indent=2)
    print(f"✅ Detailed confusion matrices saved to: {cm_json_path}")
    
    # Create summary statistics
    summary_stats = {
        'metric': ['accuracy', 'precision', 'recall', 'f1', 'auc', 'overall_score'],
        'mean': [
            df_main['accuracy'].mean(),
            df_main['precision'].mean(),
            df_main['recall'].mean(),
            df_main['f1'].mean(),
            df_main['auc'].mean(),
            df_main['overall_score'].mean()
        ],
        'std': [
            df_main['accuracy'].std(),
            df_main['precision'].std(),
            df_main['recall'].std(),
            df_main['f1'].std(),
            df_main['auc'].std(),
            df_main['overall_score'].std()
        ],
        'min': [
            df_main['accuracy'].min(),
            df_main['precision'].min(),
            df_main['recall'].min(),
            df_main['f1'].min(),
            df_main['auc'].min(),
            df_main['overall_score'].min()
        ],
        'max': [
            df_main['accuracy'].max(),
            df_main['precision'].max(),
            df_main['recall'].max(),
            df_main['f1'].max(),
            df_main['auc'].max(),
            df_main['overall_score'].max()
        ]
    }
    
    summary_csv_path = os.path.join(model_dir, 'model_evaluation_summary.csv')
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"✅ Summary statistics saved to: {summary_csv_path}")
    
    # Print summary
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"{'='*50}")
    for i, metric in enumerate(summary_stats['metric']):
        print(f"{metric.upper():12}: {summary_stats['mean'][i]:.4f} ± {summary_stats['std'][i]:.4f}")
        print(f"{'':12}  Range: [{summary_stats['min'][i]:.4f}, {summary_stats['max'][i]:.4f}]")
    
    # Create comprehensive report
    create_comprehensive_report(model_performances, model_dir)

def create_comprehensive_report(model_performances, model_dir):
    """Create a comprehensive HTML report with all results"""
    print(f"\nCreating comprehensive HTML report...")
    
    # Find best and worst performing models
    best_model = max(model_performances, key=lambda x: x['overall_score'])
    worst_model = min(model_performances, key=lambda x: x['overall_score'])
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
            .best {{ background-color: #d4edda; border-color: #c3e6cb; }}
            .worst {{ background-color: #f8d7da; border-color: #f5c6cb; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .confusion-matrix {{ display: inline-block; margin: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 Model Evaluation Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total models evaluated: {len(model_performances)}</p>
        </div>
        
        <div class="section">
            <h2>🏆 Best Performing Model</h2>
            <div class="metric best">
                <strong>Fold {best_model['fold']}</strong><br>
                Overall Score: {best_model['overall_score']:.4f}<br>
                Accuracy: {best_model['accuracy']:.4f}<br>
                F1: {best_model['f1']:.4f}<br>
                AUC: {best_model['auc']:.4f}
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Performance Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Std</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
    """
    
    # Add metrics to HTML
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'overall_score']
    for metric in metrics:
        values = [perf[metric] for perf in model_performances]
        html_content += f"""
                <tr>
                    <td>{metric.upper()}</td>
                    <td>{np.mean(values):.4f}</td>
                    <td>{np.std(values):.4f}</td>
                    <td>{np.min(values):.4f}</td>
                    <td>{np.max(values):.4f}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>🔍 Detailed Results by Fold</h2>
    """
    
    # Add detailed results for each fold
    for perf in sorted(model_performances, key=lambda x: x['fold']):
        html_content += f"""
            <div class="section">
                <h3>Fold {perf['fold']}</h3>
                <div class="metric">
                    <strong>Performance Metrics:</strong><br>
                    Accuracy: {perf['accuracy']:.4f}<br>
                    Precision: {perf['precision']:.4f}<br>
                    Recall: {perf['recall']:.4f}<br>
                    F1: {perf['f1']:.4f}<br>
                    AUC: {perf['auc']:.4f}<br>
                    Overall Score: {perf['overall_score']:.4f}
                </div>
                
                <div class="metric">
                    <strong>Training Info:</strong><br>
                    Total Epochs: {perf['total_epochs']}<br>
                    Best Epoch: {perf['best_epoch']}<br>
                    Model Size: {perf['model_size_mb']} MB<br>
                    Parameters: {perf['total_parameters']:,}
                </div>
                
                <div class="confusion-matrix">
                    <strong>Confusion Matrix:</strong><br>
                    <table style="width: auto;">
                        <tr><td></td><td>Predicted PASS</td><td>Predicted FAIL</td></tr>
                        <tr><td>Actual PASS</td><td>{perf['true_negatives']}</td><td>{perf['false_positives']}</td></tr>
                        <tr><td>Actual FAIL</td><td>{perf['false_negatives']}</td><td>{perf['true_positives']}</td></tr>
                    </table>
                </div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    html_path = os.path.join(model_dir, 'model_evaluation_report.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Comprehensive HTML report saved to: {html_path}")

def convert_model_to_onnx(model_path, output_path, input_shape):
    """Convert the best Keras model to ONNX format"""
    print(f"\nConverting model to ONNX format...")
    print(f"Input model: {model_path}")
    print(f"Output ONNX: {output_path}")
    
    try:
        # Load the model using our robust loader
        model = load_model_weights_only(model_path, input_shape)
        
        if model is None:
            print("❌ Could not load model for ONNX conversion")
            return False
            
        print(f"✅ Model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
        
        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=None,
            opset=11,  # Use ONNX opset 11 for better compatibility
            output_path=output_path
        )
        
        print(f"✅ Model successfully converted to ONNX!")
        print(f"ONNX model saved to: {output_path}")
        
        # Verify the ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed!")
        
        # Print model info
        print(f"\nONNX Model Info:")
        print(f"- Input name: {onnx_model.graph.input[0].name}")
        print(f"- Output name: {onnx_model.graph.output[0].name}")
        
        # Get input shape from ONNX model
        input_shape_onnx = []
        for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim:
            if dim.dim_value:
                input_shape_onnx.append(dim.dim_value)
            else:
                input_shape_onnx.append('dynamic')
        print(f"- Input shape: {input_shape_onnx}")
        
        output_shape_onnx = []
        for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim:
            if dim.dim_value:
                output_shape_onnx.append(dim.dim_value)
            else:
                output_shape_onnx.append('dynamic')
        print(f"- Output shape: {output_shape_onnx}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting model to ONNX: {e}")
        return False

def main():
    # Print TensorFlow version info
    print(f"TensorFlow version: {tf.__version__}")
    
    # Paths
    model_dir = "."  # Current directory where H5 files are saved
    data_dir = "/Users/senatarpan/Desktop/repos/sdc-testing-competition/tools/my_tool/data/raw_data"  # Your data directory
    
    # Check if model files exist
    h5_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_fold_') and f.endswith('.h5')]
    
    if not h5_files:
        print("No H5 model files found in the current directory.")
        print("Please make sure the H5 files are in the same directory as this script.")
        return
    
    print(f"Found {len(h5_files)} model files:")
    for file in sorted(h5_files):
        print(f"  - {file}")
    
    # Evaluate models and find the best one
    best_model, all_performances = evaluate_saved_models(model_dir, data_dir, num_folds=len(h5_files))
    
    if best_model is None:
        print("Could not evaluate models. Please check your data directory.")
        return
    
    # Convert best model to ONNX
    onnx_output_path = f"best_model_fold_{best_model['fold']}.onnx"
    
    # Get input shape from test data (since we know it's (50, 4) from the error)
    input_shape = (50, 4)  # (sequence_length, features)
    
    success = convert_model_to_onnx(
        best_model['model_path'], 
        onnx_output_path, 
        input_shape
    )
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"Best model (Fold {best_model['fold']}) has been converted to ONNX format.")
        print(f"ONNX file: {onnx_output_path}")
        print(f"\nYou can now use this ONNX model for inference in various frameworks!")
        
        # Create a summary file
        summary = {
            'best_model_info': best_model,
            'all_model_performances': all_performances,
            'onnx_file': onnx_output_path,
            'input_shape': input_shape
        }
        
        with open('model_evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📄 Evaluation summary saved to: model_evaluation_summary.json")
        
        # Print summary of all generated files
        print(f"\n📁 ALL GENERATED FILES:")
        print(f"{'='*50}")
        print(f"📊 Main results: model_evaluation_results.csv")
        print(f"🔍 Confusion matrices: confusion_matrices.csv")
        print(f"📈 Detailed confusion matrices: confusion_matrices_detailed.json")
        print(f"📋 Summary statistics: model_evaluation_summary.csv")
        print(f"🌐 HTML report: model_evaluation_report.html")
        print(f"📄 JSON summary: model_evaluation_summary.json")
        print(f"🤖 ONNX model: {onnx_output_path}")
        print(f"\n💡 You can open the HTML report in your browser for a visual overview!")

if __name__ == "__main__":
    main()