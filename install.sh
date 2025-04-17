#!/bin/bash

# Set environment name (change if needed)
ENV_NAME="genomic_data_rep"

# Create the environment from the YAML file
echo "Creating conda environment: $ENV_NAME"
conda env create -f environment.yml -n $ENV_NAME

echo "Done. Activate with: conda activate $ENV_NAME"
