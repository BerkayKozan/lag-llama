import argparse
from argument_parser import ArgumentParser
from data_loader import DataLoader
from wandb_handler import WandbHandler
from callbacks_handler import CallbacksHandler
from lag_llama_trainer import LagLlamaTrainer
from model_evaluator import ModelEvaluator
from gluonts.dataset.pandas import PandasDataset
from constants import (
    DEFAULT_DATA_PATH,
    UNNORMALIZED_BEST_CHECKPOINT_PATH,
    ZERO_SHOT_CKPT_PATH,
    NORMALIZED_BEST_CKPT_PATH,
    FORECAST_COST_PARAMS,
    EVALUATION_PARAMS,
    MAIN_DEVICE,
    PREVENTIVE_COST_DEFAULT,
    ESTIMATOR_PARAMS
)
from deterioration_probability_calculator import DeteriorationProbabilityCalculator
import wandb
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import islice
import os
import pandas as pd
from datetime import datetime
import forecast_cost_estimator
from lag_llama.gluon.estimator import LagLlamaEstimator
import torch
import joblib
import numpy as np

class Main:
    def __init__(self):
        self.data_path = DEFAULT_DATA_PATH
        self.args = ArgumentParser().args
        self.scaler = StandardScaler()
        self.data_loader = DataLoader(self.data_path, self.scaler)
        self.wandb_handler = WandbHandler(self.args)
        self.dataset = None
        self.experiment_name = self.args.experiment_name
        self.base_results_dir = self.args.results_dir

        self.time_to_repair = None
        self.probability_calculator = None
        
    def incremental_evaluation(self, predictor, evaluator, frequency):
        
        cp_cc_ratios = np.logspace(-2, np.log10(0.99), 21)  # 21 values between 0.01 and ~0.99

        test_data = self.dataset['test_df']
        related_components = test_data['item_id'].dropna().unique()
        test_data = test_data[test_data["item_id"].isin(related_components)]
        components = test_data["item_id"].unique()

        component_lengths = {
            component: len(test_data[test_data["item_id"] == component])
            for component in components
        }
        max_length = max(component_lengths.values())

        # Instead of a single list, we'll have a dictionary of time_to_repair arrays, one for each ratio scenario.
        time_to_repair_dict = {ratio: [max_length + 1] * len(components) for ratio in cp_cc_ratios}

        # Start the incremental evaluation
        for step in range(1, max_length // frequency + 2):
            # Collect slices for this step
            incremental_slices = []
            available_components = []
            for component in components:
                component_data = test_data[test_data["item_id"] == component]
                slice_end = min(step * frequency, len(component_data))
                if step * frequency - slice_end < frequency:
                    incremental_slices.append(component_data.iloc[:slice_end])
                    available_components.append(component)

            # Combine slices for all components
            combined_slice = pd.concat(incremental_slices, ignore_index=True)

            test_dataset = PandasDataset.from_long_dataframe(
                combined_slice,
                target="target",
                item_id="item_id",
            )

            # Use the predictor to forecast
            forecasts, tss, agg_metrics, ts_metrics = evaluator.evaluate(
                test_dataset,
                self.predictor,
                self.args.normalize,
                num_samples=ESTIMATOR_PARAMS["num_parallel_samples"]
            )

            self.tss = tss

            # Calculate deterioration probabilities once per step, without p_threshold
            self.deterioration_calculator = DeteriorationProbabilityCalculator(
                forecasts,
                frequency,
                max_deterioration=FORECAST_COST_PARAMS["max_deterioration_value"]
            )
            deterioration_probabilities = self.deterioration_calculator.calculate_deterioration_probability()
            print(f"Step {step} | Deterioration probabilities: {deterioration_probabilities}")

            # Now apply different p_threshold scenarios (cp_cc_ratios) after we have the probabilities
            # For each scenario ratio, if probability > ratio, we register the repair step for that scenario
            for ratio in cp_cc_ratios:
                for idx, comp_data in deterioration_probabilities.items():
                    prob_list = comp_data.get('probabilities', [])
                    for (threshold_idx, prob_value) in prob_list:
                        # If the probability surpasses this ratio scenario (interpreting ratio as a threshold)
                        if prob_value > ratio:
                            component_name = available_components[idx]
                            comp_index = components.tolist().index(component_name)
                            # Set the repair step if not already set
                            if time_to_repair_dict[ratio][comp_index] == max_length + 1:
                                time_to_repair_dict[ratio][comp_index] = (step - 1) * frequency
                            break  # Move to next component after setting repair step

            # Log metrics
            wandb.log({f"Step_{step}/Metrics": agg_metrics})

            # Plot forecasts
            self.plot_forecasts(forecasts, tss, name=f"Step_{step}")

        # After completing all steps, we have a dictionary of time_to_repair lists for each scenario.
        # Compute M_hat for each scenario
        cp = PREVENTIVE_COST_DEFAULT
        M_hat_values = []

        for ratio in cp_cc_ratios:
            cc = cp / ratio if ratio != 0 else float('inf')
            T_R_list = time_to_repair_dict[ratio]
            forecast_estimator = forecast_cost_estimator.ForecastCostEstimator(
                tss=tss,
                frequency=FORECAST_COST_PARAMS["frequency"],
                max_deterioration=FORECAST_COST_PARAMS["max_deterioration_value"],
                preventive_cost=cp,
                corrective_cost=cc,
                T_R_list=T_R_list
            )
            M_hat = forecast_estimator.calculate_costs()
            M_hat_values.append(M_hat)

        print("M_hat values:", M_hat_values)

        # Plot with normal (linear) scale
        plt.plot(cp_cc_ratios, M_hat_values, marker='o')
        plt.xlabel('cp/cc ratio')
        plt.ylabel('M_hat')
        plt.title('M_hat vs. cp/cc ratios (Linear Scale)')
        wandb.log({"Plot/M_hat_vs_Cost_linear": wandb.Image(plt.gcf())})
        plt.close()

        # Plot with log scale
        plt.plot(cp_cc_ratios, M_hat_values, marker='o')
        plt.xscale('log')  # Set the x-axis to a logarithmic scale
        plt.xlabel('cp/cc ratio (log scale)')
        plt.ylabel('M_hat')
        plt.title('M_hat vs. cp/cc ratios (Log Scale)')
        wandb.log({"Plot/M_hat_vs_Cost_log": wandb.Image(plt.gcf())})
        plt.close()

        # Save cp_cc_ratios and M_hat_values as a table
        cpcc_mhat_table = wandb.Table(columns=["cp/cc", "M_hat"])
        for ratio, m_hat in zip(cp_cc_ratios, M_hat_values):
            cpcc_mhat_table.add_data(ratio, m_hat)
        wandb.log({"Table/cpcc_vs_Mhat": cpcc_mhat_table})

        # Save time_to_repair for each scenario
        # We'll store one table that has columns for each ratio scenario
        repair_steps_table_columns = ["Component_Index"] + [f"Repair_Steps_{r}" for r in cp_cc_ratios]
        repair_steps_table = wandb.Table(columns=repair_steps_table_columns)
        for i, comp in enumerate(components):
            row = [i] + [time_to_repair_dict[r][i] for r in cp_cc_ratios]
            repair_steps_table.add_data(*row)

        wandb.log({"Repair_Steps_Table": repair_steps_table})
        
    def plot_forecasts(self, forecasts, tss, prediction_length=FORECAST_COST_PARAMS["frequency"], name="default"):
        """Generates and logs forecast plots to Wandb."""
        plt.figure(figsize=(20, 15))
        date_formatter = mdates.DateFormatter('%b, %d')
        plt.rcParams.update({'font.size': 15})

        for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
            ax = plt.subplot(3, 3, idx + 1)
            
            # Plot actual values
            plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="target")
            
            # Plot forecast
            forecast.plot(color='r')
            plt.xticks(rotation=60)
            ax.xaxis.set_major_formatter(date_formatter)
            ax.set_title(forecast.item_id)

        wandb.log({f"Plot/Forecasts/{name}": wandb.Image(plt.gcf())})

        plt.gcf().tight_layout()
        plt.legend()
        # Save the plot to a temporary file and log to WandB
        plt.close()   
   
    def setup_experiment_dirs(self):
        """
        Setup directory structure for the current run, including sweep_id and run_id.
        """
        # Get sweep ID and run ID
        run_id = wandb.run.id if wandb.run.id else datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_id = wandb.run.sweep_id if wandb.run.sweep_id else "default"


        # Create the directory structure: results/<experiment_name>/<sweep_id>/<run_id>/
        self.fulldir_experiments = os.path.join(self.base_results_dir, self.experiment_name, sweep_id, run_id)

        if os.path.exists(self.fulldir_experiments):
            print(self.fulldir_experiments, "already exists.")
        os.makedirs(self.fulldir_experiments, exist_ok=True)

        # Create directory for checkpoints
        self.checkpoint_dir = os.path.join(self.fulldir_experiments, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize CallbacksHandler with the checkpoint directory
        self.callbacks_handler = CallbacksHandler(self.args, self.checkpoint_dir)

        print(f"Experiment directories and callback handler set up at: {self.fulldir_experiments}")
    
    def train(self):
        wandb.init(project=self.args.wandb_project)
        
        # Setup directories and initialize CallbacksHandler
        self.setup_experiment_dirs()

        estimator_params = ESTIMATOR_PARAMS.copy()
        estimator_params["trainer_kwargs"]["callbacks"] = self.callbacks_handler.callbacks
        estimator_params["trainer_kwargs"]["logger"] = self.wandb_handler.logger

        # Override estimator params by wandb.config, if they exist
        for key, value in wandb.config.items():
            estimator_params[key] = value

        # Train the model
        trainer = LagLlamaTrainer(estimator_params)
        self.predictor = trainer.train(training_data=self.dataset['train'], validation_data=self.dataset['val'])
        
        wandb.finish()

    def evaluate(self, enable_normalize):
        # Load the trained predictor
        ckpt_path = UNNORMALIZED_BEST_CHECKPOINT_PATH
        non_neg_pred_samples = True
        if enable_normalize:
            ckpt_path = NORMALIZED_BEST_CKPT_PATH
            non_neg_pred_samples = False
        if self.args.zero_shot:
            ckpt_path = ZERO_SHOT_CKPT_PATH
        ckpt = torch.load(ckpt_path, map_location=MAIN_DEVICE)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        estimator_args["ckpt_path"] = ckpt_path
        estimator_args["device"] = torch.device(MAIN_DEVICE)
        estimator_args["prediction_length"] = FORECAST_COST_PARAMS["frequency"]
        trainer = LagLlamaTrainer(estimator_args)
        self.predictor = trainer.get_lag_llama_predictions(ckpt_path=estimator_args["ckpt_path"], prediction_length=estimator_args["prediction_length"], context_length=EVALUATION_PARAMS["context_length"], num_samples=EVALUATION_PARAMS["num_samples"], device=estimator_args["device"], batch_size=ESTIMATOR_PARAMS["batch_size"], nonnegative_pred_samples=non_neg_pred_samples, use_rope_scaling=self.args.rope_scaling)
        
        wandb.init(project=self.args.wandb_project)
        evaluator = ModelEvaluator(scaler=self.scaler)

        # Incremental evaluation
        self.incremental_evaluation(self.predictor, evaluator, frequency=FORECAST_COST_PARAMS["frequency"])

        wandb.finish()

    def execute_train(self, enable_normalize):
        self.data_loader.preprocess_data()
        train_df, val_df, test_df = self.data_loader.split_data()
        if enable_normalize:
            train_df, val_df, test_df, self.scaler = self.data_loader.normalize_data(train_df, val_df, test_df)
        
        # Save the scaler
        dataset_name = os.path.splitext(os.path.basename(self.data_loader.file_path))[0]
        scaler_name = f"scaler_{dataset_name}.pkl"
        scaler_path = os.path.join("scaler", scaler_name)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

        if not os.path.exists(scaler_path):
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
        else:
            print(f"Scaler already exists at {scaler_path}, skipping save.")

        dataset = {
            'train': PandasDataset.from_long_dataframe(train_df, target="target", item_id="item_id"),
            'val': PandasDataset.from_long_dataframe(val_df, target="target", item_id="item_id"),
            'test': PandasDataset.from_long_dataframe(test_df, target="target", item_id="item_id"),
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df
        }
        self.dataset = dataset

        # Run the sweep for training only
        self.wandb_handler.run_sweep(self.train, count=2)

    def execute_evaluate(self, enable_normalize):
        self.data_loader.preprocess_data()
        _, _, test_df = self.data_loader.split_data()

        # Load the scaler based on the dataset name
        dataset_name = os.path.splitext(os.path.basename(self.data_loader.file_path))[0]
        scaler_name = f"scaler_{dataset_name}.pkl"
        scaler_path = os.path.join("scaler", scaler_name)

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Please ensure the scaler is saved during training.")
        
        print(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        self.scaler = scaler
        # Normalize the test data using the preloaded scaler
        if enable_normalize:
            test_df = self.data_loader.normalize_with_scaler(test_df, scaler)

        non_nan = test_df.dropna()
        nan = test_df[test_df.isna()]
        # Concatenate non-NaNs followed by NaNs
        test_df = non_nan
        
        dataset = {
            'test': PandasDataset.from_long_dataframe(test_df, target="target", item_id="item_id"),
            'test_df': test_df
        }
        self.dataset = dataset

        # Run the evaluation
        self.evaluate(enable_normalize)
    
if __name__ == "__main__":

    main = Main()
    main.args.zero_shot = main.args.zero_shot == "True"
    main.args.rope_scaling = main.args.rope_scaling == "True"
    main.args.normalize = main.args.normalize == "True"
    enable_normalize = main.args.normalize
    print(f"Enable normalize: {enable_normalize}")
    print(f"Zero shot learning: {main.args.zero_shot}")
    print(f"Rope scaling: {main.args.rope_scaling}")
    if main.args.mode == "train":
        main.execute_train(enable_normalize)
    elif main.args.mode == "evaluate":
        main.execute_evaluate(enable_normalize)