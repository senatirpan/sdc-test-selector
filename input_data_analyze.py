import json
import statistics
from collections import Counter

def analyze_test_data(filename):
    """
    Analyze test data from JSON file to extract statistics about tests and road points
    """
    # Read the JSON data from file
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return
    
    # Initialize counters and lists for analysis
    total_tests = len(data)
    total_points = 0
    points_per_test = []
    test_outcomes = []
    test_durations = []
    rf_values = []
    oob_values = []
    max_speeds = []
    
    # Process each test record
    for i, test in enumerate(data):
        # Count road points for this test
        road_points_count = len(test.get('road_points', []))
        points_per_test.append(road_points_count)
        total_points += road_points_count
        
        # Extract metadata
        meta_data = test.get('meta_data', {})
        test_info = meta_data.get('test_info', {})
        run_config = meta_data.get('run_config', {})
        
        # Collect test outcomes and durations
        outcome = test_info.get('test_outcome')
        duration = test_info.get('test_duration')
        
        if outcome:
            test_outcomes.append(outcome)
        if duration:
            test_durations.append(duration)
        
        # Collect run configuration parameters
        rf = run_config.get('rf')
        oob = run_config.get('oob')
        max_speed = run_config.get('max_speed')
        
        if rf is not None:
            rf_values.append(rf)
        if oob is not None:
            oob_values.append(oob)
        if max_speed is not None:
            max_speeds.append(max_speed)
    
    # Calculate statistics
    print("=== TEST DATA ANALYSIS RESULTS ===")
    print(f"Total number of tests: {total_tests}")
    print(f"Total number of road points across all tests: {total_points}")
    print()
    
    # Road points statistics
    if points_per_test:
        print("=== ROAD POINTS STATISTICS ===")
        print(f"Average points per test: {statistics.mean(points_per_test):.2f}")
        print(f"Median points per test: {statistics.median(points_per_test)}")
        print(f"Minimum points per test: {min(points_per_test)}")
        print(f"Maximum points per test: {max(points_per_test)}")
        print(f"Standard deviation: {statistics.stdev(points_per_test):.2f}")
        print()
    
    # Test outcomes analysis
    if test_outcomes:
        print("=== TEST OUTCOMES ===")
        outcome_counts = Counter(test_outcomes)
        for outcome, count in outcome_counts.items():
            percentage = (count / len(test_outcomes)) * 100
            print(f"{outcome}: {count} tests ({percentage:.1f}%)")
        print()
    
    # Test duration statistics
    if test_durations:
        print("=== TEST DURATION STATISTICS ===")
        print(f"Average duration: {statistics.mean(test_durations):.2f} seconds")
        print(f"Median duration: {statistics.median(test_durations):.2f} seconds")
        print(f"Minimum duration: {min(test_durations):.2f} seconds")
        print(f"Maximum duration: {max(test_durations):.2f} seconds")
        print(f"Standard deviation: {statistics.stdev(test_durations):.2f} seconds")
        print()
    
    # Run configuration analysis
    print("=== RUN CONFIGURATION PARAMETERS ===")
    
    if rf_values:
        rf_counts = Counter(rf_values)
        print("RF (Road Friction) values:")
        for rf, count in rf_counts.items():
            percentage = (count / len(rf_values)) * 100
            print(f"  {rf}: {count} tests ({percentage:.1f}%)")
    
    if oob_values:
        oob_counts = Counter(oob_values)
        print("OOB (Out of Bounds) values:")
        for oob, count in oob_counts.items():
            percentage = (count / len(oob_values)) * 100
            print(f"  {oob}: {count} tests ({percentage:.1f}%)")
    
    if max_speeds:
        max_speed_counts = Counter(max_speeds)
        print("Max Speed values:")
        for speed, count in max_speed_counts.items():
            percentage = (count / len(max_speeds)) * 100
            print(f"  {speed}: {count} tests ({percentage:.1f}%)")
    
    print()
    
    # Sample test structure
    if data:
        print("=== SAMPLE TEST STRUCTURE ===")
        sample_test = data[0]
        print("Keys in each test record:")
        for key in sample_test.keys():
            print(f"  - {key}")
        
        if 'meta_data' in sample_test:
            print("Keys in meta_data:")
            for key in sample_test['meta_data'].keys():
                print(f"    - {key}")
        
        if 'road_points' in sample_test and sample_test['road_points']:
            print("Keys in each road point:")
            for key in sample_test['road_points'][0].keys():
                print(f"    - {key}")

# Function to analyze data distribution across different configurations
def analyze_configuration_combinations(data):
    """
    Analyze unique combinations of run configuration parameters
    """
    print("\n=== CONFIGURATION COMBINATIONS ===")
    
    config_combinations = {}
    
    for test in data:
        run_config = test.get('meta_data', {}).get('run_config', {})
        rf = run_config.get('rf')
        oob = run_config.get('oob')
        max_speed = run_config.get('max_speed')
        
        config_key = (rf, oob, max_speed)
        if config_key in config_combinations:
            config_combinations[config_key] += 1
        else:
            config_combinations[config_key] = 1
    
    print(f"Total unique configuration combinations: {len(config_combinations)}")
    print("\nConfiguration breakdown (RF, OOB, Max_Speed):")
    for config, count in sorted(config_combinations.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(data)) * 100
        print(f"  {config}: {count} tests ({percentage:.1f}%)")

# Main execution
if __name__ == "__main__":
    filename = "/Users/senatarpan/Desktop/repos/sdc-testing-competition/evaluator/sample_tests/sdc-test-data.json"  # filename path
    
    # Since the provided data appears to be a JSON array, let's handle it
    print("Analyzing test data...")
    analyze_test_data(filename)
    
    # Also load data for configuration analysis
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        analyze_configuration_combinations(data)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not perform configuration analysis: {e}")