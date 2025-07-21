#!/usr/bin/env python3
"""
Dask-like groupby simulation without Dask dependency
Simulates customer analytics with power-law distribution
"""

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time

def process_group(data_chunk):
    """Process a chunk of data for groupby operation"""
    print(f"Processing chunk with {len(data_chunk)} rows")
    start_time = time.time()
    
    # Simulate processing time based on data size
    processing_time = len(data_chunk) / 100000  # Scale factor
    time.sleep(processing_time)
    
    result = data_chunk.groupby('k')['v'].sum()
    
    end_time = time.time()
    print(f"Chunk processed in {end_time - start_time:.2f}s, groups: {len(result)}")
    
    return result

def setup_dask_test():
    """Setup skewed data for Dask-like workload"""
    print("Setting up Dask groupby test...")
    
    # Create skewed data: 99 small groups + 1 hot group
    data = pd.DataFrame({
        'k': np.concatenate([np.arange(99), np.repeat(999, 500_000)]),
        'v': 1
    })
    
    print(f"Created dataset with {len(data)} rows")
    print(f"Hot key (999) has {(data['k'] == 999).sum()} rows")
    print(f"Other keys have ~{(data['k'] != 999).sum() / 99:.0f} rows each on average")
    
    return data

if __name__ == '__main__':
    data = setup_dask_test()
    
    # Split into chunks for parallel processing (simulating Dask partitions)
    print("\nSplitting data into 2 chunks for parallel processing...")
    chunks = np.array_split(data, 2)
    
    print(f"Chunk 1: {len(chunks[0])} rows")
    print(f"Chunk 2: {len(chunks[1])} rows")
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(process_group, chunks))
    
    end_time = time.time()
    
    # Combine results
    final_result = pd.concat(results).groupby(level=0).sum()
    
    print(f"\nGroupby operation complete!")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Final groups: {len(final_result)}")
    print(f"Hot key (999) total: {final_result.get(999, 0)}")
    print(f"Sample of other keys: {dict(list(final_result.drop(999, errors='ignore').head(5).items()))}")