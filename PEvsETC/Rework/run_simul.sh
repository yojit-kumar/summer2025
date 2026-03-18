#!/bin/bash

# --- Configuration ---
ENV_NAME="etc" 
PYTHON_SCRIPT="main.py"

# 1. Initialize Conda
# This 'source' command is required because 'conda activate' 
# often doesn't work directly in shell scripts.
# Adjust the path below if your conda is installed elsewhere (e.g., /opt/anaconda3/etc...)
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null

# 2. Activate the Environment
echo "Activating environment: $ENV_NAME"
conda activate $ENV_NAME

# Check if activation worked
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate Conda environment '$ENV_NAME'"
    exit 1
fi

# 3. Run the Python Script
echo "Running Python script: $PYTHON_SCRIPT"
echo "------------------------------------------"

# python -u (unbuffered) ensures logs print immediately to the log file
python -u $PYTHON_SCRIPT

# 4. Completion
if [ $? -eq 0 ]; then
    echo "------------------------------------------"
    echo "Job Completed Successfully at $(date)"
else
    echo "------------------------------------------"
    echo "Job Failed with errors at $(date)"
    exit 1
fi
