# ECG Processing and Training

This folder contains code for model training.

## Files

- `CustomDataset.py`, `resnet.py`  
  Python modules defining dataset handling and ResNet model architecture used separately if needed.

- `train.py`  
  Script for training the model via terminal, accepting hyperparameters as arguments.

## Usage
Run training from the terminal with optional arguments:

- `--learning_rate`: Learning rate for optimizer (default 1e-3)  
- `--batch_size`: Batch size for training (default 32)  
- `--epochs`: Number of training epochs (default 10)  

Modify these as needed to tune the training process.

---

This setup allows modular use of the dataset and model code, with configurable training from the command line.
