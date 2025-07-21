#!/usr/bin/env python3
"""
Spark-like skewed workload simulation without Spark dependency
Simulates the hot key problem in distributed processing
"""

import multiprocessing as mp
import time
import random

def process_partition(partition_data):
    """Process a partition of data with skewed processing time"""
    key_count = partition_data
    
    # Simulate processing time proportional to data size
    # Each 1000 keys = ~0.01 seconds of processing
    processing_time = key_count / 100000
    
    print(f"Processing partition with {key_count} keys (estimated {processing_time:.2f}s)")
    time.sleep(processing_time)
    
    # Simulate some computation
    result = sum(range(min(key_count, 1000)))  # Cap computation to avoid memory issues
    
    return result

def setup_spark_test():
    """Setup test data for Spark-like workload"""
    print("Setting up Spark skew test...")
    
    # 99 small partitions + 1 huge partition (100x skew)
    partitions = [1000] * 99 + [1000000]  # 1K vs 1M keys
    
    print(f"Created {len(partitions)} partitions")
    print(f"Small partitions: {partitions[:-1].count(1000)} with {partitions[0]} keys each")
    print(f"Large partition: 1 with {partitions[-1]} keys")
    
    return partitions

if __name__ == '__main__':
    partitions = setup_spark_test()
    
    print("\nStarting parallel processing with 2 workers...")
    start_time = time.time()
    
    with mp.Pool(2) as pool:
        results = pool.map(process_partition, partitions)
    
    end_time = time.time()
    
    print(f"\nProcessing complete!")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Processed {len(results)} partitions")
    print(f"Total result sum: {sum(results)}")