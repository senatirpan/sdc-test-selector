import json
import numpy as np
import os
import math
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Bidirectional, LSTM, 
                                   Dense, Dropout, Flatten, BatchNormalization,
                                   GlobalMaxPooling1D, Concatenate, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
# Fix: Import specific metrics
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import tf2onnx
import onnx


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


def load_and_preprocess_data_three_features(data_dir, max_files=None, min_sequence_length=10):
    """Load JSON files and extract three features"""
    print("Loading and preprocessing data with three features (angles, lengths, curvature)...")
    
    X_angles = []
    X_lengths = []
    X_curvatures = []
    y = []
    
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json') and f != 'ReadMe.txt']
    json_files.sort()
    
    if max_files:
        json_files = json_files[:max_files]
    
    print(f"Found {len(json_files)} JSON files to process")
    
    processed = 0
    skipped = 0
    
    for i, file in enumerate(json_files):
        if i % 500 == 0:
            print(f"Processing file {i+1}/{len(json_files)}: {file}")
        
        file_path = os.path.join(data_dir, file)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            road_coordinates = data.get('road_points', [])
            test_outcome = data.get('test_outcome', 'PASS')
            
            if not road_coordinates:
                skipped += 1
                continue
            
            label_binary = 1 if test_outcome == 'FAIL' else 0
            
            # Clean road coordinates
            clean_coordinates = []
            for point in road_coordinates:
                if isinstance(point, list) and len(point) >= 2:
                    clean_coordinates.append([float(point[0]), float(point[1])])
            
            # Only process if we have enough points
            if len(clean_coordinates) >= min_sequence_length:
                angles, lengths, curvatures = extract_three_features(clean_coordinates)
                
                if len(angles) > 0 and len(lengths) > 0 and len(curvatures) > 0:
                    X_angles.append(angles)
                    X_lengths.append(lengths)
                    X_curvatures.append(curvatures)
                    y.append(label_binary)
                    processed += 1
                else:
                    skipped += 1
            else:
                skipped += 1
                
        except Exception as e:
            if i < 5:
                print(f"Error processing {file}: {e}")
            skipped += 1
            continue
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed} files")
    print(f"Skipped: {skipped} files")
    print(f"Total samples loaded: {len(X_angles)}")
    
    if y:
        pass_count = sum(1 for label in y if label == 0)
        fail_count = sum(1 for label in y if label == 1)
        print(f"Label distribution - PASS: {pass_count}, FAIL: {fail_count}")
    
    return X_angles, X_lengths, X_curvatures, y


def pad_sequences_robust(sequences, max_length=None, padding='post'):
    """Robust sequence padding with better handling"""
    if not sequences:
        return np.array([]), 0
        
    if max_length is None:
        max_length = max(len(seq) for seq in sequences if len(seq) > 0)
    
    # Cap max_length to reasonable value
    max_length = min(max_length, 500)  # Prevent memory issues
    
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


def create_three_feature_hybrid_model(input_shape, num_cnn_filters=128, lstm_units=64):
    """
    Create hybrid 1D CNN + BiLSTM model for three features
    
    Architecture Explanation:
    1. Input Layer: Receives sequences of shape (sequence_length, 3_features)
    
    2. Multi-scale CNN Feature Extraction:
       - Scale 1 (Fine-grained): kernel_size=3 for capturing local patterns
       - Scale 2 (Medium-scale): kernel_size=5 for medium-range dependencies  
       - Scale 3 (Large-scale): kernel_size=7 for long-range patterns
       Each scale has its own convolutional layers with batch normalization
    
    3. Feature Concatenation: Combines multi-scale CNN features
    
    4. Bidirectional LSTM Layers:
       - First BiLSTM: Captures temporal dependencies in both directions
       - Second BiLSTM: Further processes sequential patterns
       - Returns sequences for first layer, final states for second layer
    
    5. Global Pooling Path: Parallel extraction of global features from CNN
    
    6. Feature Fusion: Combines LSTM outputs with global pooling features
    
    7. Dense Classification Layers:
       - Multiple dense layers with regularization (L1+L2)
       - Batch normalization and dropout for preventing overfitting
       - Final sigmoid output for binary classification
    
    This architecture captures both:
    - Local geometric patterns (CNN with multiple scales)
    - Temporal road trajectory patterns (BiLSTM)
    - Global road characteristics (Global pooling)
    """
    
    # Input layer for 3 features
    inputs = Input(shape=input_shape)
    
    # Multi-scale CNN feature extraction
    # Scale 1: Fine-grained patterns (kernel=3)
    conv1 = Conv1D(filters=num_cnn_filters, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(filters=num_cnn_filters//2, kernel_size=3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    
    # Scale 2: Medium-scale patterns (kernel=5)
    conv2 = Conv1D(filters=num_cnn_filters//2, kernel_size=5, activation='relu', padding='same')(inputs)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(filters=num_cnn_filters//4, kernel_size=5, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    
    # Scale 3: Large-scale patterns (kernel=7)
    conv3 = Conv1D(filters=num_cnn_filters//4, kernel_size=7, activation='relu', padding='same')(inputs)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    
    # Concatenate multi-scale features
    concat = Concatenate()([pool1, pool2, pool3])
    concat = Dropout(0.3)(concat)
    
    # Bidirectional LSTM layers for temporal pattern recognition
    lstm1 = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2, 
                              recurrent_dropout=0.2))(concat)
    lstm1 = BatchNormalization()(lstm1)
    
    lstm2 = Bidirectional(LSTM(lstm_units//2, return_sequences=False, dropout=0.2, 
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
    
    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(dense2)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_three_feature_model_kfold(data_dir, n_splits=10, random_state=42, max_files=None):
    """Main training function with k-fold cross-validation for three features"""
    
    # Load and preprocess data
    X_angles, X_lengths, X_curvatures, y = load_and_preprocess_data_three_features(
        data_dir, max_files=max_files)
    
    if len(X_angles) == 0:
        print("No data loaded. Please check your data directory and file format.")
        return
    
    # Pad sequences
    X_angles_padded, max_len_angles = pad_sequences_robust(X_angles)
    X_lengths_padded, max_len_lengths = pad_sequences_robust(X_lengths)
    X_curvatures_padded, max_len_curvatures = pad_sequences_robust(X_curvatures)
    
    # Ensure all features have the same sequence length
    max_len = max(max_len_angles, max_len_lengths, max_len_curvatures)
    
    def pad_to_max_len(arr, current_len, target_len):
        if current_len < target_len:
            return np.pad(arr, ((0, 0), (0, target_len - current_len)), 
                         mode='constant', constant_values=0)
        return arr
    
    X_angles_padded = pad_to_max_len(X_angles_padded, max_len_angles, max_len)
    X_lengths_padded = pad_to_max_len(X_lengths_padded, max_len_lengths, max_len)
    X_curvatures_padded = pad_to_max_len(X_curvatures_padded, max_len_curvatures, max_len)
    
    # Stack features (3 features: angles, lengths, curvatures)
    X = np.stack([X_angles_padded, X_lengths_padded, X_curvatures_padded], axis=-1)
    y = np.array(y)
    
    print(f"Final data shape: {X.shape}")
    print(f"Sequence length: {max_len}")
    print(f"Number of features: {X.shape[2]} (angles, lengths, curvature)")
    print(f"Labels distribution - PASS: {np.sum(y == 0)}, FAIL: {np.sum(y == 1)}")
    
    # K-fold cross-validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = []
    fold_histories = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold + 1}/{n_splits}")
        print(f"{'='*50}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize features for this fold
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_val_reshaped = X_val.reshape(-1, X_val.shape[-1])
        
        X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
        
        # Calculate class weights for this fold
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        print(f"Fold {fold + 1} - Train: {len(X_train)}, Val: {len(X_val)}")
        print(f"Class weights: {class_weight_dict}")
        
        # Create model
        model = create_three_feature_hybrid_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Fix: Use proper metric objects instead of strings
        model.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
            loss='binary_crossentropy',
            metrics=[BinaryAccuracy(name='accuracy'), 
                    Precision(name='precision'), 
                    Recall(name='recall')]
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True,
            mode='min'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.3, 
            patience=8, 
            min_lr=0.00001,
            mode='min'
        )
        
        checkpoint = ModelCheckpoint(
            f'best_model_fold_{fold + 1}.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=150,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Evaluate
        y_val_pred_proba = model.predict(X_val_scaled)
        y_val_pred = (y_val_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        auc = roc_auc_score(y_val, y_val_pred_proba)
        
        fold_result = {
            'fold': fold + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_val, y_val_pred),
            'scaler': scaler  # Save scaler for inference
        }
        
        fold_results.append(fold_result)
        fold_histories.append(history)
        fold_models.append({'model': model, 'scaler': scaler})
        
        print(f"\nFold {fold + 1} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Confusion Matrix:\n{fold_result['confusion_matrix']}")
    
    # Find best performing fold
    best_fold_idx = np.argmax([r['f1'] for r in fold_results])  # Using F1-score as metric
    best_fold_result = fold_results[best_fold_idx]
    best_model_info = fold_models[best_fold_idx]
    
    print(f"\n{'='*60}")
    print(f"BEST PERFORMING FOLD: {best_fold_result['fold']}")
    print(f"{'='*60}")
    print(f"Best Fold F1-Score: {best_fold_result['f1']:.4f}")
    print(f"Best Fold Accuracy: {best_fold_result['accuracy']:.4f}")
    print(f"Best Fold AUC: {best_fold_result['auc']:.4f}")
    
    # Save best model as ONNX
    try:
        print("\nSaving best model as ONNX...")
        # Create a sample input for ONNX conversion
        sample_input = tf.TensorSpec(shape=(None, X.shape[1], X.shape[2]), dtype=tf.float32)
        
        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            best_model_info['model'],
            input_signature=[sample_input],
            opset=13
        )
        
        # Save ONNX model
        with open('best_model.onnx', 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        print("✓ Best model saved as 'best_model.onnx'")
        
        # Save the scaler for the best model
        import joblib
        joblib.dump(best_model_info['scaler'], 'best_model_scaler.pkl')
        print("✓ Best model scaler saved as 'best_model_scaler.pkl'")
        
    except Exception as e:
        print(f"Error saving ONNX model: {e}")
        print("Saving best model as .h5 instead...")
        best_model_info['model'].save('best_model_backup.h5')
    
    # Calculate average results
    avg_results = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results]),
        'auc': np.mean([r['auc'] for r in fold_results])
    }
    
    print(f"\n{'='*60}")
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Average Accuracy:  {avg_results['accuracy']:.4f} ± {np.std([r['accuracy'] for r in fold_results]):.4f}")
    print(f"Average Precision: {avg_results['precision']:.4f} ± {np.std([r['precision'] for r in fold_results]):.4f}")
    print(f"Average Recall:    {avg_results['recall']:.4f} ± {np.std([r['recall'] for r in fold_results]):.4f}")
    print(f"Average F1-Score:  {avg_results['f1']:.4f} ± {np.std([r['f1'] for r in fold_results]):.4f}")
    print(f"Average AUC:       {avg_results['auc']:.4f} ± {np.std([r['auc'] for r in fold_results]):.4f}")
    
    print(f"\n✓ All fold models saved as 'best_model_fold_X.h5' files")
    
    return fold_results, avg_results, best_fold_result


def preprocess_inference_data(road_points_data):
    """
    Preprocess inference data to match training format
    Handles the new JSON format with 'x' and 'y' keys
    """
    processed_data = []
    
    for sample in road_points_data:
        road_points = sample.get('road_points', [])
        
        # Convert from {"x": val, "y": val} format to [x, y] format
        clean_coordinates = []
        for point in road_points:
            if isinstance(point, dict) and 'x' in point and 'y' in point:
                clean_coordinates.append([float(point['x']), float(point['y'])])
            elif isinstance(point, list) and len(point) >= 2:
                clean_coordinates.append([float(point[0]), float(point[1])])
        
        if len(clean_coordinates) >= 3:  # Need at least 3 points for features
            angles, lengths, curvatures = extract_three_features(clean_coordinates)
            
            if len(angles) > 0 and len(lengths) > 0 and len(curvatures) > 0:
                processed_data.append({
                    'id': sample.get('_id', {}).get('$oid', 'unknown'),
                    'angles': angles,
                    'lengths': lengths,
                    'curvatures': curvatures
                })
    
    return processed_data


# Example inference function
def predict_road_outcomes(inference_file_path, model_path='best_model.onnx', scaler_path='best_model_scaler.pkl'):
    """
    Predict outcomes for new road data
    """
    import joblib
    import onnxruntime as ort
    
    # Load the inference data
    with open(inference_file_path, 'r') as f:
        inference_data = json.load(f)
    
    # Preprocess the data
    processed_data = preprocess_inference_data(inference_data)
    
    if not processed_data:
        print("No valid data found for inference")
        return []
    
    # Extract features
    X_angles = [item['angles'] for item in processed_data]
    X_lengths = [item['lengths'] for item in processed_data]
    X_curvatures = [item['curvatures'] for item in processed_data]
    
    # Pad sequences (use same max_length as training - you'll need to save this value)
    X_angles_padded, _ = pad_sequences_robust(X_angles, max_length=500)  # Use training max_length
    X_lengths_padded, _ = pad_sequences_robust(X_lengths, max_length=500)
    X_curvatures_padded, _ = pad_sequences_robust(X_curvatures, max_length=500)
    
    # Stack features
    X = np.stack([X_angles_padded, X_lengths_padded, X_curvatures_padded], axis=-1)
    
    # Load scaler and scale features
    scaler = joblib.load(scaler_path)
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_reshaped).reshape(X.shape)
    
    # Load ONNX model and predict
    ort_session = ort.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    predictions = ort_session.run(None, {input_name: X_scaled.astype(np.float32)})[0]
    
    # Format results
    results = []
    for i, item in enumerate(processed_data):
        pred_prob = predictions[i][0]
        pred_outcome = "FAIL" if pred_prob > 0.5 else "PASS"
        results.append({
            'id': item['id'],
            'predicted_outcome': pred_outcome,
            'confidence': float(pred_prob)
        })
    
    return results


if __name__ == "__main__":
    # Set data directory path
    data_dir = "/Users/senatarpan/Desktop/repos/sdc-testing-competition/tools/my_dl_tool/data/raw_data"
    
    # Use full dataset for maximum performance
    max_files = None  # Use all 10,000 files
    
    print("="*80)
    print("TRAINING 3-FEATURE HYBRID CNN+BiLSTM MODEL")
    print("="*80)
    print(f"Training with {'all 10,000' if max_files is None else 'first ' + str(max_files)} files...")
    print("\nFeatures used:")
    print("1. Turning Angles - Angular changes in road trajectory")
    print("2. Segment Lengths - Distance between consecutive road points") 
    print("3. Curvature - Road curvature at each point")
    print("\nModel improvements:")
    print("- K-fold cross-validation (10 folds)")
    print("- Multi-scale CNN architecture (3, 5, 7 kernel sizes)")
    print("- Bidirectional LSTM for temporal patterns")
    print("- Feature fusion (CNN + LSTM + Global pooling)")
    print("- Class weighting for imbalanced data")
    print("- Robust scaling and regularization")
    print("- AdamW optimizer with weight decay")
    
    # Train the model
    fold_results, avg_results, best_fold_result = train_three_feature_model_kfold(data_dir, max_files=max_files)