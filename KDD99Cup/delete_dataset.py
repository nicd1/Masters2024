import os
import shutil

# Path to the directory where the KDD99Cup dataset is stored
dataset_dir = os.path.expanduser('~/scikit_learn_data/kddcup99')

# Check if the directory exists
if os.path.exists(dataset_dir):
    # Remove the directory and all its contents
    shutil.rmtree(dataset_dir)
    print(f'Deleted the directory: {dataset_dir}')
else:
    print(f'Directory not found: {dataset_dir}')
