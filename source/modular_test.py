import forecast_cost_estimator
import pickle

with open("forecasts.pkl", "rb") as f:
    forecasts = pickle.load(f)

# Assuming forecasts is a list of SampleForecast objects and each has the `samples` array of shape (num_samples, forecast_length)

# Initialize with preventive cost, corrective cost, max deterioration threshold, and frequency
forecast_cost_estimator = forecast_cost_estimator.ForecastCostEstimator(
    forecasts=forecasts,
    frequency=5,
    max_deterioration=25,
    preventive_cost=100,
    corrective_cost=500
)

# Calculate cost metrics
M_hat, deterioration_probabilities = forecast_cost_estimator.calculate_costs()
print("PdM Policy Metric (M_hat):", M_hat)
print("Deterioration Probabilities:", deterioration_probabilities)