#!/usr/bin/env python3
"""
Flink-like join simulation without Flink dependency
Simulates retail analytics with popular items creating join skew
"""

import multiprocessing as mp
import time
import random

def process_join_partition(partition_data):
    """Process a join partition with skewed data"""
    key, values = partition_data
    
    print(f"Processing join for key {key} with {len(values)} values")
    start_time = time.time()
    
    # Simulate join processing time based on data size
    processing_time = len(values) / 10000  # Scale factor
    time.sleep(processing_time)
    
    # Simulate join operation
    result = (key, sum(values), len(values))
    
    end_time = time.time()
    print(f"Key {key} processed in {end_time - start_time:.2f}s")
    
    return result

def setup_join_data():
    """Setup skewed join data simulating retail analytics"""
    print("Setting up Flink join test...")
    
    # Create skewed data: 99 regular products + 1 popular product
    data = {}
    
    # Regular products with normal transaction volume
    for i in range(99):
        data[i] = list(range(1000))  # 1000 transactions each
    
    # One popular product (iPhone, bestseller book, etc.) with 100x more transactions
    data[999] = list(range(100000))  # Hot key with 100k transactions
    
    print(f"Created join data with {len(data)} keys")
    print(f"Regular keys: {len(data) - 1} keys with ~1000 values each")
    print(f"Hot key (999): 1 key with {len(data[999])} values")
    
    return data

if __name__ == '__main__':
    data = setup_join_data()
    
    print("\nStarting parallel join processing with 2 workers...")
    start_time = time.time()
    
    # Process join partitions in parallel
    with mp.Pool(2) as pool:
        results = pool.map(process_join_partition, data.items())
    
    end_time = time.time()
    
    # Analyze results
    print(f"\nJoin operation complete!")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Processed {len(results)} keys")
    
    # Show key statistics
    total_values = sum(result[2] for result in results)
    hot_key_result = next((r for r in results if r[0] == 999), None)
    
    print(f"Total values joined: {total_values}")
    if hot_key_result:
        print(f"Hot key (999) result: sum={hot_key_result[1]}, count={hot_key_result[2]}")
    
    # Show processing distribution
    regular_keys = [r for r in results if r[0] != 999]
    if regular_keys:
        regular_counts = [r[2] for r in regular_keys]
        print(f"Regular keys - Avg count: {sum(regular_counts)/len(regular_counts):.0f}")
        print(f"Skew ratio: {hot_key_result[2] / (sum(regular_counts)/len(regular_counts)):.1f}x" if hot_key_result else "N/A")