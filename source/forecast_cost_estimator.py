from deterioration_probability_calculator import DeteriorationProbabilityCalculator
from cost_estimator import CostEstimator

class ForecastCostEstimator:
    def __init__(self, forecasts, frequency, max_deterioration, preventive_cost, corrective_cost):
        self.deterioration_calculator = DeteriorationProbabilityCalculator(forecasts, frequency, max_deterioration)
        self.cost_estimator = CostEstimator(preventive_cost, corrective_cost, frequency)

    def calculate_costs(self):
        # Calculate deterioration probabilities
        self.deterioration_calculator.calculate_deterioration_probability()

        # Determine failure times for each forecast sample
        failure_times = self.deterioration_calculator.get_sample_failure_times()
        # Lists to store preventive replacement times and actual failure times
        T_R_list = []
        T_F_list = []

        for forecast_failure_times in failure_times:
            for T_F in forecast_failure_times:
                T_R = self.cost_estimator.calculate_perfect_replacement_time(T_F)
                T_R_list.append(T_R)
                T_F_list.append(T_F)

        # Calculate and return PdM metric
        M_hat = self.cost_estimator.calculate_metric(T_R_list, T_F_list)
        return M_hat, self.deterioration_calculator.deterioration_probabilities