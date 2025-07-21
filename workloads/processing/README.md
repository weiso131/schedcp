# Long-tail Workload Testing Framework

This framework provides automated testing for workloads with long-tail characteristics to demonstrate the benefits of custom Linux kernel schedulers.

## Files Structure

```
/home/yunwei37/schedCP/workloads/processing/
├── README.md                    # This file
├── test_cases.json             # Test case definitions
├── evaluate_workloads.py       # Main evaluation framework
├── install_deps.sh             # Dependency installer
├── case.md                     # Detailed workload documentation
└── assets/                     # Test scripts and assets
    ├── spark_skew_test.py      # Spark-like skewed workload simulation
    ├── dask_groupby_test.py    # Dask-like groupby simulation
    ├── pandas_etl_test.py      # Pandas ETL simulation
    └── flink_join_test.py      # Flink-like join simulation
```

## Quick Start

### 1. Install Dependencies

```bash
# Install required packages
sudo ./install_deps.sh
```

### 2. List Available Tests

```bash
python3 evaluate_workloads.py --list
```

### 3. Run a Single Test

```bash
# Run a specific test case
python3 evaluate_workloads.py --test spark_shuffle

# Run without process monitoring (if psutil unavailable)
python3 evaluate_workloads.py --test spark_shuffle --no-monitor
```

### 4. Run All Tests

```bash
# Run all test cases
python3 evaluate_workloads.py --all

# Save results to custom file
python3 evaluate_workloads.py --all --save my_results.json
```

## Test Cases

The framework includes 10 test cases across different categories:

### File Processing
- **pigz_compression**: Parallel compression with mixed file sizes
- **file_checksum**: Parallel checksumming with size imbalance

### Media Processing  
- **ffmpeg_transcode**: Video transcoding with one large file

### Software Testing
- **ctest_suite**: Test suite with slow integration test

### Version Control
- **git_compression**: Git garbage collection with mixed objects

### Data Processing
- **log_processing**: Log processing with skewed chunks
- **spark_shuffle**: Analytics with hot key problem
- **dask_groupby**: Customer analytics with power-law distribution
- **pandas_etl**: ETL with DDoS spike simulation
- **flink_join**: Retail analytics with popular items

## Expected Results

All test cases demonstrate the "long-tail problem" where:
- 99 tasks complete quickly (~6 seconds each)
- 1 task takes much longer (~600 seconds)
- Expected scheduler optimization: **25-35% improvement**

## Framework Features

### Process Monitoring
- Automatically detects long-running processes (>0.5s)
- Measures CPU usage and runtime distribution
- Calculates skew ratios to identify optimization potential

### Dependency Management
- Automatic dependency checking before test execution
- Graceful fallback when optional packages unavailable
- Clear error messages for missing dependencies

### Result Analysis
- JSON output with detailed timing and process data
- Summary statistics across all tests
- Long-tail detection and analysis

### Test Configuration
All test cases are defined in `test_cases.json` with:
- Setup commands (data generation)
- Test command (main workload)
- Cleanup commands (resource cleanup)
- Expected performance characteristics
- Dependencies and metadata

## Usage Examples

### Running Specific Categories

```bash
# Run only data processing tests
python3 evaluate_workloads.py --test spark_shuffle
python3 evaluate_workloads.py --test dask_groupby
python3 evaluate_workloads.py --test pandas_etl
python3 evaluate_workloads.py --test flink_join
```

### Custom Test Execution

```bash
# Run with custom timeout and save results
python3 evaluate_workloads.py --test pigz_compression --save pigz_results.json
```

### Analyzing Results

The framework saves results in JSON format with:
- Test execution times
- Process monitoring data
- Long-tail detection results
- Performance improvement estimates

Example result structure:
```json
{
  "test_id": "spark_shuffle",
  "status": "success",
  "test_time": 10.58,
  "expected_improvement": 0.33,
  "process_analysis": {
    "long_tail_detected": true,
    "skew_ratio": 100.0
  }
}
```

## Troubleshooting

### Missing Dependencies
Run the dependency installer:
```bash
sudo ./install_deps.sh
```

### Process Monitoring Issues
If psutil is unavailable, use `--no-monitor`:
```bash
python3 evaluate_workloads.py --test TEST_ID --no-monitor
```

### Permission Issues
Some tests require sudo for system commands. Run with appropriate permissions.

### Test Failures
Check the error output in the JSON results for specific failure reasons.

## Customization

### Adding New Test Cases
1. Edit `test_cases.json`
2. Add setup/test/cleanup commands
3. Specify dependencies and expected characteristics
4. Create any required asset scripts in `assets/`

### Modifying Data Sizes
Adjust the commands in `test_cases.json` to change:
- File sizes (dd commands)
- Record counts (seq commands) 
- Processing times (sleep commands in Python scripts)

## Performance Notes

- Tests are designed for 4-CPU systems
- Data sizes optimized for quick execution (seconds to minutes)
- Maintains 100:1 skew ratio between small and large tasks
- Total framework runtime: ~5-10 minutes for all tests