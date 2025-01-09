class CostEstimator:
    """
    A class to estimate costs and calculate metrics for preventive and corrective maintenance.
    Attributes:
        preventive_cost (float): The cost of preventive maintenance.
        corrective_cost (float): The cost of corrective maintenance.
        frequency (int): The frequency of preventive maintenance.
    Methods:
        calculate_replacement_cost(T_R, T_F):
            Calculates the replacement cost based on the replacement time (T_R) and failure time (T_F).
        calculate_lifecycle_length(T_R, T_F):
            Calculates the lifecycle length based on the replacement time (T_R) and failure time (T_F).
        calculate_perfect_replacement_time(T_F):
            Calculates the perfect replacement time based on the failure time (T_F).
        calculate_metric(T_R_list, T_F_list):
            Calculates the Metric proposed by Kamariotis et al. 2023 based on lists of replacement times (T_R_list) and failure times (T_F_list).
    """
    
    def __init__(self, preventive_cost, corrective_cost, frequency):
        self.C_p = preventive_cost
        self.C_c = corrective_cost
        self.frequency = frequency

    def calculate_replacement_cost(self, T_R, T_F):
        if T_R is None or T_F is None:
            return None  # Skip invalid data
        return self.C_p if T_R < T_F else self.C_c

    def calculate_lifecycle_length(self, T_R, T_F):
        if T_R is None or T_F is None:
            return None  # Skip invalid data
        return min(T_R, T_F)

    def calculate_perfect_replacement_time(self, T_F):
        if T_F is None:
            return None  # Skip invalid data
        return (T_F // self.frequency) * self.frequency

    def calculate_metric(self, T_R_list, T_F_list):
        valid_data = [
            (T_R, T_F)
            for T_R, T_F in zip(T_R_list, T_F_list)
            if T_R is not None and T_F is not None
        ]
        
        if not valid_data:
            raise ValueError("No valid T_R and T_F data available for calculation.")

        n = len(valid_data)
        C_rep_list = [self.calculate_replacement_cost(T_R, T_F) for T_R, T_F in valid_data]
        T_lc_list = [self.calculate_lifecycle_length(T_R, T_F) for T_R, T_F in valid_data]
        T_R_perfect_list = [self.calculate_perfect_replacement_time(T_F) for _, T_F in valid_data]

        # Remove None values from lists to avoid issues in calculations
        C_rep_list = [x for x in C_rep_list if x is not None]
        T_lc_list = [x for x in T_lc_list if x is not None]
        T_R_perfect_list = [x for x in T_R_perfect_list if x is not None]

        if not (C_rep_list and T_lc_list and T_R_perfect_list):
            raise ValueError("Insufficient valid data to calculate metrics.")

        avg_C_rep = sum(C_rep_list) / len(C_rep_list)
        avg_T_lc = sum(T_lc_list) / len(T_lc_list)
        avg_T_R_perfect = sum(T_R_perfect_list) / len(T_R_perfect_list)

        M_hat = ((avg_C_rep / avg_T_lc) - (self.C_p / avg_T_R_perfect)) / (self.C_p - avg_T_R_perfect)
        return M_hat