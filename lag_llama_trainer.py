import torch
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import tqdm

class LagLlamaTrainer:
    def __init__(self, estimator_params):
        self.estimator_params = estimator_params
        self.ckpt_path = estimator_params.get("ckpt_path", "models/lag-llama.ckpt")

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
    
    def create_predictor(self):
        # Load parameters from checkpoint
        self.load_checkpoint_params()
        # Initialize LagLlamaEstimator with the updated parameters

        estimator = LagLlamaEstimator(**self.estimator_params)
        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)
        # Create the predictor
        return predictor
    
    def get_lag_llama_predictions(self, ckpt_path = "..models/lag-llama.ckpt", prediction_length=5, context_length=128, num_samples=20, device="cuda", batch_size=64, nonnegative_pred_samples=False, use_rope_scaling=False):
        ckpt = torch.load(ckpt_path, map_location=device)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        print(f"Rope Scaling: {use_rope_scaling}")
        rope_scaling_arguments = {
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        }
        estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=prediction_length,
            context_length=context_length,

            # estimator args
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],

            nonnegative_pred_samples=nonnegative_pred_samples,
            # linear positional encoding scaling
            rope_scaling=rope_scaling_arguments if use_rope_scaling else None,
            batch_size=batch_size,
            num_parallel_samples=num_samples,
            device=torch.device(device),
        )

        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, lightning_module)

        return predictor