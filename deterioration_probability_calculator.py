import numpy as np

class DeteriorationProbabilityCalculator:
    def __init__(self, forecasts, frequency, max_deterioration):
        """
        Initializes the deterioration probability calculator.
        
        :param forecasts: List of SampleForecast objects. Each object contains an array of shape (num_samples, forecast_length).
        :param frequency: Interval at which to check for deterioration (e.g., 5 for every 5th step).
        :param max_deterioration: The threshold value beyond which deterioration is considered.
        """
        self.forecasts = forecasts
        self.frequency = frequency
        self.max_deterioration = max_deterioration
        self.deterioration_probabilities = []
        
    def calculate_deterioration_probability(self):
        """
        Calculates the probability of deterioration over time for each component based on forecast samples.
        For each component, determines the first time step where the probability exceeds the critical percentage.

        :param critical_percentage: The percentage threshold to decide when to repair.
        :return: A dictionary mapping each component to its repair step and probability timeline.
        """
        # Dictionary to store results for each component
        component_results = {}

        for comp_idx, forecast in enumerate(self.forecasts):  # Iterate over components
            num_samples, forecast_length = forecast.samples.shape
            print("num_samples: ", num_samples)
            print("forecast_length: ", forecast_length)
            probabilities = []  # To store probabilities at each time step
            repair_step = None  # To store the first step exceeding critical threshold

            for t in range(0, forecast_length, self.frequency):
                # For each timestep, count how many samples exceed max_deterioration
                count_deteriorated = (forecast.samples > self.max_deterioration).any(axis=1).sum()

                # Calculate probability as a percentage
                deterioration_probability = (count_deteriorated / num_samples)
                probabilities.append((t, deterioration_probability))

            # Store the results for this component
            component_results[comp_idx] = {
                "probabilities": probabilities,
            }
        return component_results

    def get_results(self):
        """
        Returns the calculated deterioration probabilities as a list of (timestep, probability) tuples.
        """
        if not self.deterioration_probabilities:
            self.calculate_deterioration_probability()
        return self.deterioration_probabilities
    
    def get_sample_failure_times(self):
        """
        Determines the first failure time for each sample based on max_deterioration threshold.
        
        :return: List of failure times for each sample in each forecast.
        """
        failure_times = []
        for forecast in self.forecasts:
            forecast_failure_times = []
            for sample in forecast.samples:
                try:
                    T_F = np.where(sample > self.max_deterioration)[0][0]  # First time it exceeds threshold
                except IndexError:
                    T_F = None  # If never exceeds, set to end of forecast
                forecast_failure_times.append(T_F)
            failure_times.append(forecast_failure_times)
        return failure_times