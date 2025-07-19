#!/usr/bin/env python3
"""
Validate schedulers.json against the schema and convert to new format if needed.
"""

import json
import sys
import os
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data, filepath):
    """Save JSON file with proper formatting."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def convert_old_to_new_format(old_scheduler):
    """Convert old scheduler format to new simplified format."""
    name = list(old_scheduler.keys())[0]
    old_data = old_scheduler[name]
    
    # Extract use cases from workload_types
    use_cases = old_data.get('workload_types', [])
    
    # Convert characteristics from dict/list to string
    characteristics = old_data.get('characteristics', {})
    if isinstance(characteristics, dict):
        char_items = [k.replace('_', ' ') for k, v in characteristics.items() if v is True]
        characteristics_str = ', '.join(char_items) if char_items else "No specific characteristics"
    elif isinstance(characteristics, list):
        characteristics_str = ', '.join(characteristics)
    else:
        characteristics_str = str(characteristics)
    
    # Convert tuning parameters
    tuning_params = {}
    old_params = old_data.get('tuning_parameters', {})
    for param_name, param_info in old_params.items():
        new_param = {
            'type': 'string',  # Default type
            'description': param_info.get('description', ''),
            'default': param_info.get('default', '')
        }
        
        # Infer type from default value or range
        if 'range' in param_info:
            if isinstance(param_info['range'], str) and '-' in param_info['range']:
                try:
                    min_val, max_val = param_info['range'].split('-')
                    new_param['type'] = 'integer'
                    new_param['range'] = [int(min_val), int(max_val)]
                except:
                    pass
            elif isinstance(param_info['range'], list):
                new_param['type'] = 'float' if any('.' in str(x) for x in param_info['range']) else 'integer'
                new_param['range'] = param_info['range']
        
        # Check default value type
        default = param_info.get('default', '')
        if default in ['true', 'false']:
            new_param['type'] = 'boolean'
            new_param['default'] = default == 'true'
        elif default.isdigit():
            new_param['type'] = 'integer'
            new_param['default'] = int(default)
        elif '.' in default and default.replace('.', '').isdigit():
            new_param['type'] = 'float'
            new_param['default'] = float(default)
        
        tuning_params[param_name] = new_param
    
    # Build new format
    new_scheduler = {
        'name': name,
        'production_ready': old_data.get('production_ready', False),
        'description': old_data.get('description', 'No description available')[:100],
        'use_cases': use_cases if use_cases else ['general_purpose'],
        'algorithm': old_data.get('scheduling_algorithm', old_data.get('type', 'unknown')),
        'characteristics': characteristics_str,
        'tuning_parameters': tuning_params
    }
    
    # Add optional fields if present
    if 'characteristics' in old_data and 'nohz_full_requirement' in str(old_data['characteristics']):
        new_scheduler['requirements'] = "Linux kernel with nohz_full support"
    
    # Extract limitations from selection_guide cons
    if 'selection_guide' in old_data and 'cons' in old_data['selection_guide']:
        cons = old_data['selection_guide']['cons']
        if isinstance(cons, list):
            new_scheduler['limitations'] = ', '.join(cons[:2])  # Take first 2 limitations
    
    # Create performance profile from selection guide
    if 'selection_guide' in old_data:
        guide = old_data['selection_guide']
        perf_hints = []
        if 'latency' in str(guide).lower():
            perf_hints.append('optimized for low latency')
        if 'throughput' in str(guide).lower():
            perf_hints.append('good throughput')
        if perf_hints:
            new_scheduler['performance_profile'] = ', '.join(perf_hints)
    
    return new_scheduler

def validate_and_convert():
    """Validate schedulers.json against schema and convert if needed."""
    scheduler_dir = Path(__file__).parent
    schema_path = scheduler_dir / 'json' / 'scheduler-schema.json'
    schedulers_path = scheduler_dir / 'schedulers.json'
    
    # Load schema
    try:
        schema = load_json(schema_path)
    except Exception as e:
        print(f"Error loading schema: {e}")
        return 1
    
    # Load schedulers
    try:
        data = load_json(schedulers_path)
    except Exception as e:
        print(f"Error loading schedulers.json: {e}")
        return 1
    
    # Check if it's in old format (nested structure)
    if 'schedulers' in data and isinstance(data['schedulers'], dict):
        print("Detected old format. Converting to new format...")
        
        new_schedulers = []
        for scheduler_name, scheduler_data in data['schedulers'].items():
            new_scheduler = convert_old_to_new_format({scheduler_name: scheduler_data})
            new_schedulers.append(new_scheduler)
        
        new_data = {'schedulers': new_schedulers}
        
        # Save backup
        backup_path = schedulers_path.with_suffix('.json.backup')
        save_json(data, backup_path)
        print(f"Backup saved to {backup_path}")
        
        # Save new format
        save_json(new_data, schedulers_path)
        print(f"Converted and saved to {schedulers_path}")
        
        data = new_data
    
    # Validate against schema
    try:
        validate(instance=data, schema=schema)
        print("✓ schedulers.json is valid according to the schema")
        return 0
    except ValidationError as e:
        print(f"✗ Validation error: {e.message}")
        print(f"  Path: {' -> '.join(str(p) for p in e.path)}")
        return 1

if __name__ == '__main__':
    sys.exit(validate_and_convert())