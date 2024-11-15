import torch
from lag_llama.gluon.estimator import LagLlamaEstimator

class LagLlamaTrainer:
    def __init__(self, estimator_params):
        self.estimator_params = estimator_params
        self.ckpt_path = estimator_params.get("ckpt_path", "lag-llama.ckpt")

    def load_checkpoint_params(self):
        # Load the checkpoint and extract required parameters
        ckpt = torch.load(self.ckpt_path, map_location=self.estimator_params["device"])
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        
        # Extract specific parameters from the checkpoint
        self.estimator_params.update({
            "input_size": estimator_args["input_size"],
            "n_layer": estimator_args["n_layer"],
            "n_embd_per_head": estimator_args["n_embd_per_head"],
            "n_head": estimator_args["n_head"],
            "time_feat": estimator_args["time_feat"]
        })

    def train(self, training_data, validation_data):
        # Load parameters from checkpoint
        self.load_checkpoint_params()
        # Initialize LagLlamaEstimator with the updated parameters
        estimator = LagLlamaEstimator(**self.estimator_params)

        # Train the estimator with training and validation data
        predictor = estimator.train(training_data=training_data, validation_data=validation_data)

        return predictor