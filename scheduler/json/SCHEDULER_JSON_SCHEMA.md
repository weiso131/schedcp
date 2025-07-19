# Scheduler JSON Schema Documentation

This document defines the standardized JSON schema for scheduler definitions in the AI-OS project.

## Overview

The scheduler JSON provides a consistent format for describing available schedulers, their capabilities, configuration options, and use cases. This enables automated tooling and consistent documentation.

## Schema Structure

### Root Object
```json
{
  "schedulers": [
    // Array of scheduler objects
  ]
}
```

### Scheduler Object

Each scheduler object must contain the following fields:

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique identifier for the scheduler (e.g., "scx_bpfland") |
| `production_ready` | boolean | Whether the scheduler is ready for production use |
| `description` | string | Brief one-line description (max 100 characters) |
| `use_cases` | array[string] | Primary use cases for this scheduler |
| `algorithm` | string | Core scheduling algorithm used |
| `characteristics` | string | Key features and behaviors description |
| `tuning_parameters` | object | Configuration parameters (see below) |

#### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `requirements` | string | System requirements (kernel version, hardware, etc.) |
| `limitations` | string | Known limitations or constraints |
| `performance_profile` | string | Expected performance characteristics and behavior |

### Tuning Parameters Schema

Each parameter in `tuning_parameters` should follow this structure:

```json
"parameter_name": {
  "type": "integer|float|boolean|string",
  "default": <default_value>,
  "range": [min, max],  // For numeric types
  "options": ["opt1", "opt2"],  // For string enums
  "description": "Clear description of the parameter"
}
```


## Example

```json
{
  "schedulers": [
    {
      "name": "scx_rusty",
      "production_ready": true,
      "description": "Multi-domain scheduler with intelligent load balancing for general-purpose workloads",
      "use_cases": ["general_purpose", "multi_socket", "mixed_workloads"],
      "algorithm": "multi_domain_round_robin",
      "characteristics": "Multi-domain scheduler with per-LLC domains, intelligent load balancing, and flexible architecture support",
      "tuning_parameters": {
        "slice_us": {
          "type": "integer",
          "default": 20000,
          "range": [1000, 100000],
          "description": "Time slice duration in microseconds"
        },
        "greedy_threshold": {
          "type": "integer",
          "default": 2,
          "range": [1, 10],
          "description": "Threshold for greedy task stealing"
        },
        "kthreads": {
          "type": "boolean",
          "default": false,
          "description": "Schedule kernel threads"
        }
      },
      "requirements": "Linux kernel 6.12+ with sched_ext support, Rust 1.82+",
      "limitations": "May not distinguish NUMA nodes well, potential infeasible weights issue",
      "performance_profile": "Medium latency with high throughput and low overhead, excellent for general-purpose mixed workloads"
    }
  ]
}
```

## Field Guidelines

### name
- Must match the scheduler binary name
- Use lowercase with underscores
- Example: "scx_layered"

### production_ready
- `true`: Thoroughly tested and suitable for production
- `false`: Experimental, testing, or educational schedulers

### description
- Concise, single-line description
- Focus on the primary purpose
- Maximum 100 characters

### use_cases
- List specific workload types
- Common values: "gaming", "multimedia", "batch_processing", "containers", "general_purpose"
- Be specific to help users choose

### algorithm
- Technical description of the core algorithm
- Examples: "vruntime_based", "edf", "multi_domain", "cgroup_aware"

### characteristics
- Single string describing key technical features
- Be comprehensive but concise
- Focus on what makes this scheduler unique

### tuning_parameters
- Include all user-configurable parameters
- Provide clear descriptions
- Always include default values
- Specify valid ranges or options

## Benefits

1. **Consistency**: All schedulers follow the same structure
2. **Automation**: Tools can parse and use scheduler information
3. **Documentation**: Auto-generate user guides from JSON
4. **Validation**: Ensure all required information is present
5. **Discoverability**: Users can easily compare schedulers

## Migration Notes

When updating existing scheduler definitions:
1. Remove verbose fields like `selection_guide`
2. Consolidate related fields (e.g., merge `type` and `scheduling_algorithm` into `algorithm`)
3. Move detailed explanations to separate documentation
4. Keep JSON focused on facts, not recommendations