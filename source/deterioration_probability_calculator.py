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
        Calculates the probability of deterioration over time based on forecast samples.
        """
        # Get the number of samples and forecast length from the first forecast's samples
        num_samples, forecast_length = self.forecasts[0].samples.shape

        print(f"Number of samples: {num_samples}")
        print(f"Forecast length: {forecast_length}")
        print(f"Forecasts: {self.forecasts}")
        # Iterate over each timestep with the specified frequency
        for t in range(0, forecast_length, self.frequency):
            # For each forecast, count samples exceeding max_deterioration at this timestep
            count_deteriorated = sum(
                (forecast.samples[:, t] > self.max_deterioration).sum() for forecast in self.forecasts
            )
            print(f"Count deteriorated at timestep {t}: {count_deteriorated}")
            # Calculate probability as a percentage
            deterioration_probability = (count_deteriorated / num_samples)
            self.deterioration_probabilities.append((t, deterioration_probability))

            # Stop if deterioration reaches 100%
            if deterioration_probability >= 1:
                break

        return self.deterioration_probabilities

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
                    T_F = len(sample)  # If never exceeds, set to end of forecast
                forecast_failure_times.append(T_F)
            failure_times.append(forecast_failure_times)
        return failure_times