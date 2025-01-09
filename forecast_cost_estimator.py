from deterioration_probability_calculator import DeteriorationProbabilityCalculator
from cost_estimator import CostEstimator
import numpy as np

class ForecastCostEstimator:
    def __init__(self, tss, frequency, max_deterioration, preventive_cost, corrective_cost, T_R_list):
        self.p_threshold = preventive_cost / corrective_cost
        self.tss = tss
        self.max_deterioration = max_deterioration
        self.cost_estimator = CostEstimator(preventive_cost, corrective_cost, frequency)
        self.T_R_list = T_R_list

    def calculate_costs(self):
        # Lists to store preventive replacement times and actual failure times
        T_F_list = []
        for component in self.tss:
            deteriorations = component[0]
            cleaned_deterioration = component[0][~np.isnan(component[0])]
            for idx, deterioration in enumerate(cleaned_deterioration):
                if deterioration > self.max_deterioration:
                    T_F_list.append(idx)
                    break  # Stop after finding the first exceedance
      
        # Calculate and return PdM metric
        M_hat = self.cost_estimator.calculate_metric(self.T_R_list, T_F_list)
        return M_hat
