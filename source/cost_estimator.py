class CostEstimator:
    def __init__(self, preventive_cost, corrective_cost, frequency):
        self.C_p = preventive_cost
        self.C_c = corrective_cost
        self.frequency = frequency

    def calculate_replacement_cost(self, T_R, T_F):
        return self.C_p if T_R < T_F else self.C_c

    def calculate_lifecycle_length(self, T_R, T_F):
        return min(T_R, T_F)

    def calculate_perfect_replacement_time(self, T_F):
        return (T_F // self.frequency) * self.frequency

    def calculate_metric(self, T_R_list, T_F_list):
        n = len(T_R_list)
        C_rep_list = [self.calculate_replacement_cost(T_R, T_F) for T_R, T_F in zip(T_R_list, T_F_list)]
        T_lc_list = [self.calculate_lifecycle_length(T_R, T_F) for T_R, T_F in zip(T_R_list, T_F_list)]
        T_R_perfect_list = [self.calculate_perfect_replacement_time(T_F) for T_F in T_F_list]

        avg_C_rep = sum(C_rep_list) / n
        avg_T_lc = sum(T_lc_list) / n
        avg_T_R_perfect = sum(T_R_perfect_list) / n

        M_hat = ((avg_C_rep / avg_T_lc) - (self.C_p / avg_T_R_perfect)) / (self.C_p - avg_T_R_perfect)
        return M_hat