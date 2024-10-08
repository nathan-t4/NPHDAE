#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="../plotting/compare_dc_microgrids_realistic.py"

# Define an array of values for the argument
PARAM_VALUES=(1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8)

# Loop through each value and run the Python script
for param in "${PARAM_VALUES[@]}"; do
    echo "Running $PYTHON_SCRIPT with --param $param"
    python "$PYTHON_SCRIPT" --param "$param"
done