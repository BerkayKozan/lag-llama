# model_evaluator.py

import numpy as np
from tqdm.autonotebook import tqdm
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.model.forecast import SampleForecast
import wandb
import joblib

class ModelEvaluator:
    def __init__(self, scaler):
        self.scaler = scaler
        self.evaluator = Evaluator()
    
    def evaluate(self, dataset, predictor, enable_normalize, num_samples=10):
        forecast_it, ts_it = make_evaluation_predictions(dataset=dataset, predictor=predictor, num_samples=num_samples)
        forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting"))
        tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))
        if not enable_normalize:
            agg_metrics, ts_metrics = self.evaluator(iter(tss), iter(forecasts))
            return forecasts, tss, agg_metrics, ts_metrics
        denormalized_forecasts = self.denormalize_forecasts(forecasts)
        denormalized_tss = self.denormalize_tss(tss)
        agg_metrics, ts_metrics = self.evaluator(iter(denormalized_tss), iter(denormalized_forecasts))
        return denormalized_forecasts, denormalized_tss, agg_metrics, ts_metrics
    
    def denormalize_forecasts(self, forecasts):
        denormalized_forecasts = []
        for forecast in forecasts:
            samples = forecast.samples

            # Apply the inverse transform to each sample in the forecast
            denormalized_samples = np.array([
                self.scaler.inverse_transform(sample.reshape(1, -1)).flatten()
                for sample in samples
            ])
            
            # Create a new SampleForecast object with the denormalized samples
            denormalized_forecast = SampleForecast(
                samples=denormalized_samples,
                start_date=forecast.start_date,
                item_id=forecast.item_id
            )
            denormalized_forecasts.append(denormalized_forecast)
        
        return denormalized_forecasts
    
    def denormalize_tss(self, tss):
        denormalized_tss = []
        for ts in tss:
            ts_copy = ts.copy()
            values = ts_copy.values.reshape(-1, 1)
            denormalized_values = self.scaler.inverse_transform(values)
            ts_copy.iloc[:, 0] = denormalized_values.flatten()
            denormalized_tss.append(ts_copy)
        return denormalized_tss