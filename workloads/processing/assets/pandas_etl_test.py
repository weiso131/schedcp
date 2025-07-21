#!/usr/bin/env python3
"""
Pandas ETL simulation without heavy dependencies
Simulates log processing with DDoS spike creating large file
"""

import multiprocessing as mp
import pandas as pd
import glob
import gzip
import time
import io

def parse_log_file(filename):
    """Parse a gzipped log file and return basic stats"""
    print(f"Processing {filename}...")
    start_time = time.time()
    
    try:
        with gzip.open(filename, 'rt') as f:
            lines = f.readlines()
            
        # Simulate processing time based on file size
        processing_time = len(lines) / 10000  # Scale factor
        time.sleep(processing_time)
        
        # Simple parsing simulation
        data = []
        for line in lines[:1000]:  # Limit to avoid memory issues
            parts = line.strip().split()
            if len(parts) >= 4:
                data.append({
                    'timestamp': parts[0] + ' ' + parts[1],
                    'level': parts[2],
                    'message': ' '.join(parts[3:])
                })
        
        df = pd.DataFrame(data)
        
        end_time = time.time()
        print(f"Processed {filename} in {end_time - start_time:.2f}s: {len(lines)} lines -> {len(df)} parsed")
        
        return {
            'filename': filename,
            'total_lines': len(lines),
            'parsed_lines': len(df),
            'processing_time': end_time - start_time
        }
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return {'filename': filename, 'error': str(e)}

if __name__ == '__main__':
    print("Starting Pandas ETL test...")
    
    # Find all log files
    files = glob.glob('logs/*.gz')
    print(f"Found {len(files)} log files to process")
    
    if not files:
        print("No log files found! Run setup first.")
        exit(1)
    
    start_time = time.time()
    
    # Process files in parallel
    with mp.Pool(2) as pool:
        results = pool.map(parse_log_file, files)
    
    end_time = time.time()
    
    # Combine and analyze results
    successful_results = [r for r in results if 'error' not in r]
    
    print(f"\nETL processing complete!")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Files processed: {len(successful_results)}")
    print(f"Total lines processed: {sum(r['total_lines'] for r in successful_results)}")
    print(f"Total parsed records: {sum(r['parsed_lines'] for r in successful_results)}")
    
    # Show processing time distribution
    if successful_results:
        times = [r['processing_time'] for r in successful_results]
        print(f"Processing time - Min: {min(times):.2f}s, Max: {max(times):.2f}s, Avg: {sum(times)/len(times):.2f}s")