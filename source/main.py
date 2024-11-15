from argument_parser import ArgumentParser
from data_loader import DataLoader
from wandb_handler import WandbHandler
from callbacks_handler import CallbacksHandler
from lag_llama_trainer import LagLlamaTrainer
from model_evaluator import ModelEvaluator
from gluonts.dataset.pandas import PandasDataset
from constants import ESTIMATOR_PARAMS
from deterioration_probability_calculator import DeteriorationProbabilityCalculator
import wandb
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from itertools import islice
matplotlib.use('Agg')

class Main:
    def __init__(self):
        self.args = ArgumentParser().args
        self.scaler = StandardScaler()
        self.data_loader = DataLoader("datasets/deterioration/ts_long_1.0_100_5_2.parquet", self.scaler)
        self.wandb_handler = WandbHandler(self.args)
        self.dataset = None
        # Initialize CallbacksHandler with args
        self.callbacks_handler = CallbacksHandler(self.args)
        self.probability_calculator = None

    def train_and_evaluate(self):
        wandb.init(project=self.args.wandb_project)
        estimator_params = ESTIMATOR_PARAMS.copy()
        estimator_params["trainer_kwargs"]["callbacks"] = self.callbacks_handler.callbacks
        estimator_params["trainer_kwargs"]["logger"] = self.wandb_handler.logger
        # Cosine Annealing LR
        if self.args.use_cosine_annealing_lr:
            cosine_annealing_lr_args = {"T_max": self.args.cosine_annealing_lr_t_max, "eta_min": self.args.cosine_annealing_lr_eta_min}
        else:
            cosine_annealing_lr_args = {}
        estimator_params["use_cosine_annealing_lr"] = self.args.use_cosine_annealing_lr
        estimator_params["cosine_annealing_lr_args"] = cosine_annealing_lr_args

        # Train the model
        trainer = LagLlamaTrainer(estimator_params)
        print(estimator_params)
        print(wandb.config)
        self.predictor = trainer.train(training_data=self.dataset['train'], validation_data=self.dataset['val'])

        # Evaluate after training
        evaluator = ModelEvaluator(scaler=self.scaler)

        forecasts, tss, agg_metrics, ts_metrics = evaluator.evaluate(self.dataset['test'], self.predictor)
        # Save forecasts for later testing
        with open("forecasts.pkl", "wb") as f:
            pickle.dump(forecasts, f)        
        agg_metrics = {f"Metrics/{key}": value for key, value in agg_metrics.items()}

        # Log evaluation metrics to WandB
        wandb.log(agg_metrics)

        # Calculate deterioration probabilities and log
        probability_calculator = DeteriorationProbabilityCalculator(forecasts, frequency=5, max_deterioration=25)
        probabilities = probability_calculator.calculate_deterioration_probability()
        wandb.log({"Metrics/deterioration_probabilities": probabilities})
        # Generate and log forecast plots
        self.plot_forecasts(forecasts, tss)
        wandb.finish()

    def plot_forecasts(self, forecasts, tss, prediction_length=24):
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

        wandb.log({"Plot/Forecasts": wandb.Image(plt.gcf())})
        plt.gcf().tight_layout()
        plt.legend()
        plt.show()
        # Save the plot to a temporary file and log to WandB
        plt.close()

    def execute(self):
        self.data_loader.preprocess_data()
        train_df, val_df, test_df = self.data_loader.split_data()
        train_df, val_df, test_df = self.data_loader.normalize_data(train_df, val_df, test_df)
        
        dataset = {
            'train': PandasDataset.from_long_dataframe(train_df, target="target", item_id="item_id"),
            'val': PandasDataset.from_long_dataframe(val_df, target="target", item_id="item_id"),
            'test': PandasDataset.from_long_dataframe(test_df, target="target", item_id="item_id")
        }
        self.dataset = dataset
        self.wandb_handler.run_sweep(self.train_and_evaluate, count=2)


if __name__ == "__main__":
    Main().execute()