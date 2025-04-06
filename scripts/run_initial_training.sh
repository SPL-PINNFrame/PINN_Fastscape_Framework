#!/bin/bash

# Activate your Python environment if necessary (e.g., conda activate gdalenv)
# Replace 'gdalenv' with your actual environment name if different

echo "Starting PINN training..."

# Run the training script with the specified configuration file
python PINN_Fastscape_Framework/scripts/train.py --config PINN_Fastscape_Framework/configs/initial_train_adaptive.yaml

echo "Training script finished."