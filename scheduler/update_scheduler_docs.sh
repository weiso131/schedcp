#!/bin/bash

# Script to update scheduler documentation with help output

SCHE_BIN_DIR="/root/yunwei37/ai-os/scheduler/sche_bin"
SCHE_DESC_DIR="/root/yunwei37/ai-os/scheduler/sche_description"

echo "Updating scheduler documentation with help output..."

# Get unique scheduler names (removing hash suffixes)
schedulers=$(ls "$SCHE_BIN_DIR" | grep -E '^scx_[a-z_]+$' | sort -u)

for scheduler in $schedulers; do
    echo "Processing $scheduler..."
    
    # Skip certain binaries that aren't actual schedulers
    if [[ "$scheduler" == "scx_loader" || "$scheduler" == "scx_lib_selftests" ]]; then
        echo "  Skipping $scheduler (not a scheduler)"
        continue
    fi
    
    desc_file="$SCHE_DESC_DIR/$scheduler.md"
    
    # Check if description file exists
    if [[ ! -f "$desc_file" ]]; then
        echo "  Warning: No description file found for $scheduler"
        continue
    fi
    
    # Try --help first, then -h
    help_output=""
    if $SCHE_BIN_DIR/$scheduler --help 2>&1 | grep -q "^Usage:"; then
        help_output=$($SCHE_BIN_DIR/$scheduler --help 2>&1)
    else
        # Try -h instead
        help_output=$($SCHE_BIN_DIR/$scheduler -h 2>&1)
    fi
    
    # Check if we got valid help output
    if [[ -z "$help_output" ]] || [[ "$help_output" == *"invalid option"* && ! "$help_output" == *"Usage:"* ]]; then
        echo "  Warning: Could not get help output for $scheduler"
        continue
    fi
    
    # Check if help section already exists
    if grep -q "^## Command Line Options" "$desc_file"; then
        echo "  Help section already exists in $desc_file, skipping..."
        continue
    fi
    
    # Append help output to the description file
    cat >> "$desc_file" << EOF

## Command Line Options

\`\`\`
$help_output
\`\`\`
EOF
    
    echo "  Updated $desc_file"
done

echo "Documentation update complete!"