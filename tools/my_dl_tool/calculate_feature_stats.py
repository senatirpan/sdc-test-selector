#!/usr/bin/env python3
"""
Script to calculate correct feature statistics from training data
This fixes the feature scaling mismatch between training and inference
"""

import json
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
import glob

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
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
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

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
                from scipy.interpolate import interp1d
                interpolator = interp1d(current_indices, features, kind='linear', 
                                      fill_value='extrapolate', bounds_error=False)
                interpolated = interpolator(target_indices)
                interpolated = np.nan_to_num(interpolated, nan=0.0)
                adjusted_features.append(interpolated)
    
    return adjusted_features

def calculate_feature_statistics():
    """Calculate feature statistics from training data"""
    print("🔍 Calculating feature statistics from training data...")
    
    data_dir = "data/raw_data"  # Fixed path
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    print(f"📁 Found {len(json_files)} training files")
    
    all_features = []
    processed_count = 0
    
    for i, file_path in enumerate(json_files):
        if i % 1000 == 0:
            print(f"📊 Processing file {i+1}/{len(json_files)}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            road_coordinates = data.get('road_points', [])
            if not road_coordinates:
                continue
            
            # Extract features
            angles, lengths, curvatures, velocity_changes = extract_enhanced_features(road_coordinates)
            
            if len(lengths) == 0:
                continue
            
            # Adjust to target size
            adjusted_features = adjust_features_to_target_size(
                [angles, lengths, curvatures, velocity_changes], 50
            )
            
            # Stack features
            feature_data = np.stack(adjusted_features, axis=-1)
            
            # Flatten for statistics calculation
            feature_flat = feature_data.reshape(-1, 4)
            all_features.append(feature_flat)
            
            processed_count += 1
            
        except Exception as e:
            if i < 5:  # Only show first few errors
                print(f"⚠️  Error processing {file_path}: {e}")
            continue
    
    if not all_features:
        print("❌ No features extracted!")
        return None
    
    # Combine all features
    all_features_combined = np.vstack(all_features)
    print(f"✅ Extracted features from {processed_count} files")
    print(f"📊 Total feature vectors: {all_features_combined.shape[0]}")
    print(f"🔢 Features per vector: {all_features_combined.shape[1]}")
    
    # Calculate statistics manually (like RobustScaler)
    center_ = np.median(all_features_combined, axis=0)
    q75, q25 = np.percentile(all_features_combined, [75, 25], axis=0)
    scale_ = q75 - q25
    scale_[scale_ == 0] = 1.0  # Avoid division by zero
    
    print(f"\n📊 FEATURE STATISTICS CALCULATED:")
    print(f"   📐 Angles: median={center_[0]:.4f}, IQR={scale_[0]:.4f}")
    print(f"   📏 Lengths: median={center_[1]:.4f}, IQR={scale_[1]:.4f}")
    print(f"   🔄 Curvatures: median={center_[2]:.6f}, IQR={scale_[2]:.6f}")
    print(f"   ⚡ Velocity changes: median={center_[3]:.4f}, IQR={scale_[3]:.4f}")
    
    return {
        'center_': center_,
        'scale_': scale_
    }

def update_test_selector_file(stats):
    """Update the FEATURE_STATS in my_test_selector.py"""
    if not stats:
        print("❌ No statistics to update!")
        return
    
    file_path = "my_test_selector.py"  # Fixed path
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Create the new FEATURE_STATS string
        new_stats = f"""# CRITICAL: Actual training statistics - DO NOT CHANGE!
FEATURE_STATS = {{
    'center_': np.array([{stats['center_'][0]:.6f}, {stats['center_'][1]:.6f}, {stats['center_'][2]:.8f}, {stats['center_'][3]:.6f}]),  # [angles, lengths, curvatures, velocity_changes]
    'scale_': np.array([{stats['center_'][0]:.6f}, {stats['center_'][1]:.6f}, {stats['center_'][2]:.8f}, {stats['center_'][3]:.6f}])    # Actual IQR values from training data
}}"""
        
        # Replace the old FEATURE_STATS
        import re
        pattern = r'# CRITICAL:.*?FEATURE_STATS = \{.*?\}'
        replacement = new_stats
        
        if re.search(pattern, content, re.DOTALL):
            new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            print("✅ Found and replaced FEATURE_STATS")
        else:
            # If pattern not found, replace the placeholder
            old_pattern = r'FEATURE_STATS = \{.*?\}'
            if re.search(old_pattern, content, re.DOTALL):
                new_content = re.sub(old_pattern, replacement, content, flags=re.DOTALL)
                print("✅ Found and replaced placeholder FEATURE_STATS")
            else:
                print("❌ Could not find FEATURE_STATS to replace!")
                return
        
        # Write the updated file
        with open(file_path, 'w') as f:
            f.write(new_content)
        
        print(f"✅ Successfully updated {file_path}")
        print(f"🔧 New FEATURE_STATS applied:")
        print(f"   Center: {stats['center_']}")
        print(f"   Scale: {stats['scale_']}")
        
    except Exception as e:
        print(f"❌ Error updating file: {e}")

if __name__ == "__main__":
    print("🚀 Starting feature statistics calculation...")
    
    # Calculate statistics
    stats = calculate_feature_statistics()
    
    if stats:
        # Update the test selector file
        update_test_selector_file(stats)
        
        print("\n🎯 NEXT STEPS:")
        print("1. ✅ Feature statistics calculated and updated")
        print("2. 🔄 Restart your test selector")
        print("3. 🧪 Test with new scaling values")
        print("4. 📊 Model should now predict correctly!")
    else:
        print("❌ Failed to calculate statistics!")
