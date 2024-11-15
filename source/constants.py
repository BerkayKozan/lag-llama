# constants.py
import torch

ARGUMENTS = [

    {
        "flags": ["-e", "--experiment_name"],
        "type": str,
        "help": "Name of the experiment"
    },
    {
        "flags": ["--seed"],
        "type": int,
        "default": 42,
        "help": "Random seed for reproducibility"
    },
    {
        "flags": ["--swa"],
        "action": "store_true",
        "help": "Using Stochastic Weight Averaging"
    },
    {
        "flags": ["--swa_lrs"],
        "type": float,
        "default": 1e-2,
        "help": "Learning rate for SWA"
    },
    {
        "flags": ["--swa_epoch_start"],
        "type": float,
        "default": 0.8,
        "help": "Starting epoch for SWA"
    },
    {
        "flags": ["--annealing_epochs"],
        "type": int,
        "default": 10,
        "help": "Number of annealing epochs"
    },
    {
        "flags": ["--annealing_strategy"],
        "type": str,
        "default": "cos",
        "choices": ["cos", "linear"],
        "help": "Annealing strategy"
    },
    {
        "flags": ["-b", "--batch_size"],
        "type": int,
        "default": 256,
        "help": "Batch size for training"
    },
    {
        "flags": ["-m", "--max_epochs"],
        "type": int,
        "default": 10000,
        "help": "Maximum number of epochs"
    },
    {
        "flags": ["-n", "--num_batches_per_epoch"],
        "type": int,
        "default": 100,
        "help": "Number of batches per epoch"
    },
    {
        "flags": ["--limit_val_batches"],
        "type": int,
        "help": "Limit the number of validation batches"
    },
    {
        "flags": ["--early_stopping_patience"],
        "type": int,
        "default": 2,
        "help": "Patience for early stopping"
    },
    {
        "flags": ["--dropout"],
        "type": float,
        "default": 0.0,
        "help": "Dropout rate"
    },
    {
        "flags": ["-w", "--wandb_entity"],
        "type": str,
        "default": None,
        "help": "WandB entity for logging"
    },
    {
        "flags": ["--wandb_project"],
        "type": str,
        "default": "lag-llama-test",
        "help": "WandB project name"
    },
    {
        "flags": ["--wandb_tags"],
        "nargs": "+",
        "help": "Tags for WandB"
    },
    {
        "flags": ["--wandb_mode"],
        "type": str,
        "default": "online",
        "choices": ["offline", "online"],
        "help": "WandB mode"
    },
        {
        "flags": ["--use_cosine_annealing_lr"],
        "action": "store_true",
        "default": True,
        "help": "Enable cosine annealing learning rate scheduler"
    },
    {
        "flags": ["--cosine_annealing_lr_t_max"],
        "type": int,
        "default": 10000,
        "help": "T_max value for the cosine annealing learning rate scheduler"
    },
    {
        "flags": ["--cosine_annealing_lr_eta_min"],
        "type": float,
        "default": 1e-2,
        "help": "Minimum learning rate for the cosine annealing learning rate scheduler"
    }
]

SWEEP_CONFIG = {
    "method": "bayes",  # options: 'grid', 'random', 'bayes'
    "metric": {
        "name": "mean_wQuantileLoss",
        "goal": "minimize"
    },
    "parameters": {
        "context_length": {
            "values": [20, 60, 80]  # Adjust values as needed
        },
        "factor": {
            "values": [0.1, 0.2, 0.3]
        },
        "patience": {
            "values": [1, 2, 3]
        },
    }
}

# Default values for quick import
DEFAULT_ARGS = {
    "experiment_name": "lag_llama_test",
    "enable_swa": True,
    "swa": True,
    "swa_lrs": 0.01,
    "swa_epoch_start": 0.8,
    "annealing_epochs": 1,
    "annealing_strategy": "cos",
    "enable_early_stopping": True,
    "early_stopping_patience": 50,
    "wandb_project": "lag-llama-test"
}

# Default values for the estimator parameters
ESTIMATOR_PARAMS = {
    "ckpt_path": "lag-llama.ckpt",
    "prediction_length": 10,
    "context_length": 20,
    "nonnegative_pred_samples": True,
    "aug_prob": 0,
    "lr": 5e-4,
    "dropout": 0,
    "batch_size": 32,
    "num_parallel_samples": 10,
    "trainer_kwargs": {
        "max_epochs": 5,
    },
    "device": torch.device("cpu"),
}