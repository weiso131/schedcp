# Long-tail Workload Testing Framework

This framework provides automated testing for parallel workloads with long-tail characteristics to demonstrate the benefits of custom Linux kernel schedulers. The framework uses a 40-task configuration with severe load imbalance (39 short tasks + 1 long task) to simulate real-world scenarios.

## Files Structure

```
/home/yunwei37/ai-os/workloads/processing/
├── README.md                    # This file
├── test_cases_parallel.json    # Parallel test case definitions (40 tasks: 39 short + 1 long)
├── evaluate_workloads_parallel.py  # Main parallel evaluation framework
├── install_deps.sh             # Dependency installer
├── case.md                     # Detailed workload documentation
└── assets/                     # Test scripts and assets
    ├── short.c                 # Short-running C program for testing
    ├── long.c                  # Long-running C program for testing
    ├── spark_skew_prepare.py   # Hot key aggregation data generator
    ├── spark_skew_test.py      # Hot key aggregation workload simulation
    ├── pandas_etl_prepare.py   # DDoS log analysis data generator
    ├── pandas_etl_test.py      # DDoS log analysis simulation
    ├── flink_join_prepare.py   # Viral product analytics data generator
    └── flink_join_test.py      # Viral product analytics simulation
```

## Quick Start

### 1. Install Dependencies

```bash
# Install required packages
sudo ./install_deps.sh
```

### 2. List Available Tests

```bash
python3 evaluate_workloads_parallel.py --list
```

### 3. Run a Single Test

```bash
# Run a specific test case with parallel execution
python3 evaluate_workloads_parallel.py --test hotkey_aggregation

# Run without process monitoring (if psutil unavailable)
python3 evaluate_workloads_parallel.py --test spark_shuffle --no-monitor
```

### 4. Run All Tests

```bash
# Run all test cases in parallel
python3 evaluate_workloads_parallel.py --all

# Save results to custom file
python3 evaluate_workloads_parallel.py --all --save my_results.json
```

## Test Cases

The framework includes 9 test cases across different categories, each configured with 40 parallel tasks (39 short + 1 long):

### File Processing
- **pigz_compression**: Parallel compression of mixed-size files with severe load imbalance
- **file_checksum**: Parallel file system operations with one large file blocking completion

### Media Processing  
- **ffmpeg_transcode**: Video transcoding with one large file dominating processing time

### Software Testing
- **ctest_suite**: Test suite with fast unit tests and one slow integration test

### Version Control
- **git_compression**: Git incremental compression with mixed object sizes

### Data Processing
- **log_processing**: Log processing with skewed chunks and different compression levels
- **hotkey_aggregation**: Analytics with skewed data distribution (hot key problem)
- **ddos_log_analysis**: Security log analysis with temporal spike pattern (DDoS simulation)
- **viral_product_analytics**: Retail analytics with temporal hot product pattern (trending item simulation)

## Expected Results

All test cases demonstrate the "long-tail problem" where:
- 39 short tasks complete quickly (few seconds each)
- 1 long task takes much longer (significantly more time)
- Expected scheduler optimization: **30-35% improvement**
- Total workload: 40 parallel tasks with severe imbalance

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
All test cases are defined in `test_cases_parallel.json` with:
- Separate setup commands for small and large tasks
- Small commands (executed 39 times in parallel)  
- Large commands (executed 1 time in parallel)
- Cleanup commands (resource cleanup)
- Expected performance characteristics and improvement ratios
- Dependencies and metadata

## Usage Examples

### Running Specific Categories

```bash
# Run only data processing tests
python3 evaluate_workloads_parallel.py --test hotkey_aggregation
python3 evaluate_workloads_parallel.py --test ddos_log_analysis
python3 evaluate_workloads_parallel.py --test viral_product_analytics

# Run file processing tests
python3 evaluate_workloads_parallel.py --test pigz_compression
python3 evaluate_workloads_parallel.py --test file_checksum
```

### Custom Test Execution

```bash
# Run with custom timeout and save results
python3 evaluate_workloads_parallel.py --test pigz_compression --save pigz_results.json
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
  "test_id": "hotkey_aggregation",
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
python3 evaluate_workloads_parallel.py --test TEST_ID --no-monitor
```

### Permission Issues
Some tests require sudo for system commands. Run with appropriate permissions.

### Test Failures
Check the error output in the JSON results for specific failure reasons.

## Customization

### Adding New Test Cases
1. Edit `test_cases_parallel.json`
2. Add small_setup, large_setup, small_commands, and large_commands
3. Specify dependencies and expected improvement ratios
4. Create any required asset scripts in `assets/`

### Modifying Data Sizes
Adjust the commands in `test_cases_parallel.json` to change:
- File sizes (dd commands, count parameters)
- Record counts (seq commands, loop ranges)
- Processing intensity (parameters in Python preparation scripts)

## Performance Notes

- Tests are designed for 4-CPU systems
- Configuration: 40 parallel tasks (39 short + 1 long)
- Data sizes optimized for demonstrable imbalance
- Expected improvement: 30-35% with custom schedulers
- Total framework runtime: varies by test case complexity
- Each test case specifically designed to create scheduler optimization opportunities