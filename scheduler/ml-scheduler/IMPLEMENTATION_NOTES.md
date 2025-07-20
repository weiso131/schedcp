# ML Scheduler Implementation Notes

This is an exact extraction of the ML-based scheduler from the scx_rusty fork commit 7eb71797d35d7d2f45daba7418159c5ab902d1ab.

## Key Components

### 1. TensorFlow Model Integration
- Uses TensorFlow Rust bindings (v0.21.0)
- Loads pre-trained SavedModel for migration decisions
- Model input: 5 features (cpu, cpu_idle, cpu_not_idle, src_dom_load, dst_dom_load)
- Model output: Binary decision (migrate or not)

### 2. Core Implementation
- `TensorFlowModel`: Wrapper for TensorFlow model loading and inference
- `migrate_inference()`: Exact function from the fork that makes migration decisions
- Takes CPU and load metrics as input
- Returns boolean migration decision

### 3. BPF Interface
The original implementation adds these fields to task context:
```c
/* Machine Learning inference needed data */
unsigned long src_dom_load;
unsigned long dst_dom_load;
int cpu_idle;
int cpu_not_idle;
int cpu;
```

### 4. Model Details
- Pre-trained model: `model_stress_model.h5`
- Converted to SavedModel format using `transfer.py`
- Model trained to reduce unnecessary migrations (77% reduction achieved)

## Usage

The ML scheduler is integrated into the load balancer as follows:
1. Load balancer initializes with model path
2. For each migration decision, calls `migrate_inference()`
3. Model predicts whether migration will improve performance
4. Only migrations predicted as beneficial are performed

## Performance
According to the paper and implementation:
- 10% improvement in kernel compilation time
- 77% reduction in task migrations
- Dynamic adaptation to workload patterns

## Building and Testing
```bash
# Build the scheduler
cargo build --release

# Run tests
cargo run --bin test-ml
```

## Original Paper
"Can ML Techniques Improve Linux Kernel Scheduler Decisions?" (https://arxiv.org/abs/2407.10077)