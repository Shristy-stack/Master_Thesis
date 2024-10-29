
# UNet Model for Semantic Segmentation

This repository contains a PyTorch Lightning implementation of a UNet model designed for semantic segmentation. The model supports data augmentation, training, validation, and testing with logging and checkpointing.


## Installation

Requirements
Python 3.11

PyTorch

PyTorch Lightning

Torchvision

TensorBoard

SLURM (for running on HPC clusters)

```bash
  pip install -r requirements.txt
```

Set up your environment if you're running on an HPC cluster (details in the SLURM script example below).


    
## Usage

Running the Training Script Locally
To start training the UNet model locally, run the following command:

```bash
 python train_unet.py
```
Arguments

Set up paths for your data and model hyperparameters in train_unet.py:

image_dir, mask_dir:  Training image and mask directories.

image_dir_val, mask_dir_val: Validation image and mask directories.

image_dir_test, mask_dir_test: Test image and mask directories.

num_epochs: Number of training epochs.

batch_size: Batch size for training.

learning_rate: Learning rate for the optimizer.



Running the Training Script on a SLURM Cluster
To run the model on a SLURM-based HPC cluster, use the provided batch script (train_unet.slurm).

Submit the Job:

```bash
 sbatch slurm.sh
```

SLURM Script 

```bash
 #!/bin/bash -l
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=6:00:00
#SBATCH --export=ALL  # Export all environment variables
#SBATCH --error=my_thesis/segmentation/Unet_implementation/logging/segs_syntaxs_2.error
#SBATCH --output=my_thesis/segmentation/Unet_implementation/logging/segs_syntaxs_2.output

# Navigate to the project directory
cd /home/vault/iwi5/iwi5208h/my_thesis/segmentation/Unet_implementation

# Set up Python environment to point to Python 3.11 local packages
export PYTHONUSERBASE=/home/hpc/iwi5/iwi5208h/.local
export PATH=/apps/jupyterhub/jh3.1.1-py3.11/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONPATH

# Ensure the correct Python binary is used
/apps/jupyterhub/jh3.1.1-py3.11/bin/python3 train_unet.py

```

SLURM Script Explanation:

#SBATCH --gres=gpu:a100:1:  Requests a GPU of type A100.

#SBATCH --partition=a100: Sets the partition to A100.

#SBATCH --time=6:00:00: Limits the job to a maximum runtime of 6 hours

#SBATCH --error and #SBATCH --output: Specify error and output log files

Environment Setup in the Script:

Ensure the correct PYTHONUSERBASE and PYTHONPATH are set.
Use /apps/jupyterhub/jh3.1.1-py3.11/bin/python3 to run the training script in this environment.





## Configuration

Logging: TensorBoard logs are saved in the tb_logs directory. To view them, use:

```bash
  tensorboard --logdir=tb_logs
```

Model Checkpoints: Model checkpoints are saved in the checkpoints directory. The best model is stored based on validation dice score.

Example Output Logs
Check output and error logs:

logging/segs_syntaxs_2.output: Contains standard output.

logging/segs_syntaxs_2.error: Contains error logs.






## Example


```bash
  # Run locally
python train_unet.py

# Run on SLURM
sbatch train_unet.slurm
```

This README provides instructions on setting up, training, and evaluating the UNet model, both locally and on an HPC cluster.

