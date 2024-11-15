import wandb
from datetime import datetime
from lightning.pytorch.loggers import WandbLogger
from constants import SWEEP_CONFIG

class WandbHandler:
    def __init__(self, args):
        self.args = args
        self.logger = None
        self.sweep_id = None
    

    def configure_logger(self):
        """Configure and return a WandB logger synchronized with the current wandb run."""
        experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Initialize WandbLogger with the W&B run's details
        self.logger = WandbLogger(
            project=self.args.wandb_project,
            id=experiment_id,
            mode=self.args.wandb_mode,
        )
        return self.logger
    
    def initialize_sweep(self):
        """Initializes a sweep using the SWEEP_CONFIG from constants.py"""
        if self.sweep_id is None:
            try:
                self.sweep_id = wandb.sweep(SWEEP_CONFIG, project=self.args.wandb_project)
                print(f"Sweep ID: {self.sweep_id}")
            except wandb.Error as e:
                print(f"Error initializing sweep: {e}")
                raise

    def run_sweep(self, train_function, count=5):
        """Runs a WandB sweep with a specified training function."""
        if self.sweep_id is None:
            self.initialize_sweep()

        # Ensure an active sweep ID before proceeding
        if self.sweep_id:
            # Use a wrapper to ensure the logger is configured within each sweep iteration
            def wrapped_train_function():
                # Configure logger in sync with the current wandb run
                self.configure_logger()
                train_function()

            wandb.agent(self.sweep_id, function=wrapped_train_function, count=count)
        else:
            print("Failed to start sweep: Sweep ID is None")