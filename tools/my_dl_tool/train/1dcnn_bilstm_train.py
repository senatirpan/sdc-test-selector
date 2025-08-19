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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight


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


def calculate_segment_length(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def load_and_preprocess_data_enhanced(data_dir, max_files=None, min_sequence_length=10):
    """Load JSON files and extract enhanced features"""
    print("Loading and preprocessing data with enhanced features...")
    
    X_angles = []
    X_lengths = []
    X_curvatures = []
    X_velocity_changes = []
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
                angles, lengths, curvatures, velocity_changes = extract_enhanced_features(clean_coordinates)
                
                if len(angles) > 0 and len(lengths) > 0:
                    X_angles.append(angles)
                    X_lengths.append(lengths)
                    X_curvatures.append(curvatures)
                    X_velocity_changes.append(velocity_changes)
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
    
    return X_angles, X_lengths, X_curvatures, X_velocity_changes, y


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


def create_improved_hybrid_model(input_shape, num_cnn_filters=128, lstm_units=64):
    """Create improved hybrid 1D CNN + BiLSTM model"""
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Multi-scale CNN feature extraction
    # Scale 1: Fine-grained patterns
    conv1 = Conv1D(filters=num_cnn_filters, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv1D(filters=num_cnn_filters//2, kernel_size=3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    
    # Scale 2: Medium-scale patterns
    conv2 = Conv1D(filters=num_cnn_filters//2, kernel_size=5, activation='relu', padding='same')(inputs)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv1D(filters=num_cnn_filters//4, kernel_size=5, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    
    # Scale 3: Large-scale patterns
    conv3 = Conv1D(filters=num_cnn_filters//4, kernel_size=7, activation='relu', padding='same')(inputs)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    
    # Concatenate multi-scale features
    concat = Concatenate()([pool1, pool2, pool3])
    concat = Dropout(0.3)(concat)
    
    # Bidirectional LSTM layers
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
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(dense2)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_improved_model_kfold(data_dir, n_splits=10, random_state=42, max_files=None):
    """Main training function with k-fold cross-validation"""
    
    # Load and preprocess data
    X_angles, X_lengths, X_curvatures, X_velocity_changes, y = load_and_preprocess_data_enhanced(
        data_dir, max_files=max_files)
    
    if len(X_angles) == 0:
        print("No data loaded. Please check your data directory and file format.")
        return
    
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
    
    # Stack features (4 features: angles, lengths, curvatures, velocity_changes)
    X = np.stack([X_angles_padded, X_lengths_padded, X_curvatures_padded, X_velocity_changes_padded], axis=-1)
    y = np.array(y)
    
    print(f"Final data shape: {X.shape}")
    print(f"Sequence length: {max_len}")
    print(f"Labels distribution - PASS: {np.sum(y == 0)}, FAIL: {np.sum(y == 1)}")
    
    # K-fold cross-validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = []
    fold_histories = []
    
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
        model = create_improved_hybrid_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Compile with AdamW optimizer
        model.compile(
            optimizer=AdamW(learning_rate=0.001, weight_decay=0.01),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
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
            'confusion_matrix': confusion_matrix(y_val, y_val_pred)
        }
        
        fold_results.append(fold_result)
        fold_histories.append(history)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Confusion Matrix:\n{fold_result['confusion_matrix']}")
    
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
    
    return fold_results, avg_results

if __name__ == "__main__":
    # Set data directory path
    data_dir = "/Users/senatarpan/Desktop/repos/sdc-testing-competition/tools/my_tool/data/raw_data"
    
    # Use full dataset for maximum performance
    max_files = None  # Use all 10,000 files
    
    print(f"Training improved hybrid model with {'all 10,000' if max_files is None else 'first ' + str(max_files)} files...")
    print("Improvements implemented:")
    print("K-fold cross-validation (10 folds)")
    print("Enhanced features (angles, lengths, curvature, velocity changes)")
    print("Multi-scale CNN architecture")
    print("Improved hybrid CNN+BiLSTM design")
    print("Better regularization and normalization")
    print("Class weighting for imbalanced data")
    print("Robust scaling and preprocessing")
    print("AdamW optimizer with weight decay")
    
    # Train the improved model
    fold_results, avg_results = train_improved_model_kfold(data_dir, max_files=max_files)