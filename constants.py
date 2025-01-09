# constants.py
import torch

ARGUMENTS = [

    {
        "flags": ["-e", "--experiment_name"],
        "type": str,
        "default": "lag-llama-evaluation_rope_long_cl",
        "help": "Name of the experiment"
    },
    
    {
        "flags": ["-r", "--rope"],
        "type": str,
        "default": "True",
        "help": "Enable Rope Scaling"
    },
    
    {
        "flags": ["-m", "--mode"],
        "type": str,
        "help": "Train or evaluate"
    },
    {
        "flags": ["-n", "--normalize"],
        "type": str,
        "default": "True",
        "help": "Normalize the data"
    },

    {
        "flags": ["-z", "--zero_shot"],
        "type": str,
        "default": "False",
        "help": "Zero shot learning"
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
        "default": "lag-llama-evaluation",
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
        "default": False,
        "help": "Enable cosine annealing learning rate scheduler"
    },
    {
        "flags": ["--cosine_annealing_lr_t_max"],
        "type": int,
        "default": 300,
        "help": "T_max value for the cosine annealing learning rate scheduler"
    },
    {
        "flags": ["--cosine_annealing_lr_eta_min"],
        "type": float,
        "default": 0,
        "help": "Minimum learning rate for the cosine annealing learning rate scheduler"
    },
    {
        "flags": ["--enable_model_checkpointing"],
        "action": "store_true",
        "default": True,
        "help": "Enable model checkpointing",
    },
    {
        "flags": ["--results_dir"],
        "type": str,
        "default": "results",
        "help": "Directory to store results",
    },
    {
        "flags": ["--early_stopping_patience"],
        "type": int,
        "default": 7,
        "help": "Patience for early stopping",
    },
    {
        "flags": ["--enable_early_stopping"],
        "action": "store_true",
        "default": True,
        "help": "Enable early stopping",
    }
        
]

SWEEP_CONFIG = {
    "method": "bayes",  # options: 'grid', 'random', 'bayes'
    "metric": {
        "name": "val_loss", #todo : change to val loss
        "goal": "minimize"
    },
    "parameters": {
        "context_length": {
            "values": [64, 128, 256]
        },
        "lr": {
            "distribution": "log_uniform_values",
            "max": 2e-4,
            "min": 1e-6,
        },
    }
}

FORECAST_COST_PARAMS = {
    "max_deterioration_value": 25,
    "frequency": 5,
}

# Default values for the estimator parameters
ESTIMATOR_PARAMS = {
    "ckpt_path": "models/lag-llama.ckpt",
    "prediction_length": FORECAST_COST_PARAMS["frequency"],
    "nonnegative_pred_samples": True,
    "aug_prob": 0,
    "dropout": 0,
    "batch_size": 32,
    "num_parallel_samples": 100,
    "trainer_kwargs": {
        "max_epochs": 100,
    },
    "device": torch.device("cpu"),
}

