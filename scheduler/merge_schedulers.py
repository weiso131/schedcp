#!/usr/bin/env python3
"""
Script to merge individual scheduler JSON files from sche_description directory
into a single schedulers.json file.
"""

import json
import os
import sys
from datetime import datetime

def merge_scheduler_jsons(input_dir, output_file, backup=True):
    """
    Merge individual scheduler JSON files into a single schedulers.json file.
    
    Args:
        input_dir: Directory containing individual scheduler JSON files
        output_file: Path to the output schedulers.json file
        backup: Whether to create a backup of existing output file
    """
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return False
    
    # Create backup if output file exists and backup is requested
    if backup and os.path.exists(output_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{output_file}.backup_{timestamp}"
        try:
            with open(output_file, 'r') as f:
                backup_data = f.read()
            with open(backup_file, 'w') as f:
                f.write(backup_data)
            print(f"Created backup: {backup_file}")
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
    
    # List all JSON files in the directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    json_files.sort()  # Sort for consistent ordering
    
    if not json_files:
        print(f"Error: No JSON files found in '{input_dir}'.")
        return False
    
    print(f"Found {len(json_files)} JSON files to merge.")
    
    # Collect all schedulers
    schedulers = []
    errors = []
    
    for json_file in json_files:
        file_path = os.path.join(input_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract scheduler data
            if 'scheduler' in data:
                scheduler = data['scheduler']
                schedulers.append(scheduler)
                print(f"✓ Loaded: {json_file}")
            else:
                errors.append(f"Missing 'scheduler' key in {json_file}")
                print(f"✗ Error: Missing 'scheduler' key in {json_file}")
                
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in {json_file}: {e}")
            print(f"✗ Error: Invalid JSON in {json_file}: {e}")
        except Exception as e:
            errors.append(f"Error reading {json_file}: {e}")
            print(f"✗ Error reading {json_file}: {e}")
    
    # Sort schedulers by name for consistent ordering
    schedulers.sort(key=lambda x: x.get('name', ''))
    
    # Create the merged structure
    merged_data = {
        "schedulers": schedulers
    }
    
    # Write the merged file
    try:
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully merged {len(schedulers)} schedulers into: {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False
    
    # Report any errors
    if errors:
        print(f"\nEncountered {len(errors)} errors during merge:")
        for error in errors:
            print(f"  - {error}")
    
    return True

def validate_merged_file(file_path):
    """
    Validate the merged schedulers.json file.
    
    Args:
        file_path: Path to the schedulers.json file to validate
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'schedulers' not in data:
            print("Validation Error: 'schedulers' key not found.")
            return False
        
        schedulers = data['schedulers']
        print(f"\nValidation Results:")
        print(f"  - Total schedulers: {len(schedulers)}")
        
        # Check for required fields
        missing_fields = []
        for scheduler in schedulers:
            name = scheduler.get('name', 'Unknown')
            required_fields = ['name', 'production_ready', 'description', 'use_cases', 
                             'algorithm', 'characteristics', 'tuning_parameters']
            
            for field in required_fields:
                if field not in scheduler:
                    missing_fields.append(f"{name}: missing '{field}'")
        
        if missing_fields:
            print(f"  - Missing required fields:")
            for item in missing_fields:
                print(f"    * {item}")
        else:
            print(f"  - All schedulers have required fields ✓")
        
        # List all scheduler names
        print(f"\nSchedulers in merged file:")
        for scheduler in schedulers:
            prod_status = "✓" if scheduler.get('production_ready', False) else "✗"
            print(f"  {prod_status} {scheduler.get('name', 'Unknown')}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"Validation Error: Invalid JSON - {e}")
        return False
    except Exception as e:
        print(f"Validation Error: {e}")
        return False

def main():
    """Main function to run the merge script."""
    # Define paths
    input_dir = "/root/yunwei37/ai-os/scheduler/sche_description"
    output_file = "/root/yunwei37/ai-os/scheduler/schedulers.json"
    
    # Parse command line arguments
    backup = True
    if len(sys.argv) > 1 and sys.argv[1] == "--no-backup":
        backup = False
    
    print(f"Merging scheduler JSONs from: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Backup enabled: {backup}")
    print("-" * 50)
    
    # Merge the schedulers
    if merge_scheduler_jsons(input_dir, output_file, backup):
        print("\nMerge completed successfully!")
        
        # Validate the merged file
        print("\n" + "=" * 50)
        print("Validating merged file...")
        validate_merged_file(output_file)
    else:
        print("\nMerge completed with errors.")
        sys.exit(1)

if __name__ == "__main__":
    # Print usage information
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("Usage: python3 merge_schedulers.py [--no-backup]")
        print("\nOptions:")
        print("  --no-backup    Skip creating a backup of existing schedulers.json")
        print("\nThis script merges individual scheduler JSON files from the")
        print("sche_description directory into a single schedulers.json file.")
        sys.exit(0)
    
    main()