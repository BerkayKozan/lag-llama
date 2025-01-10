import datetime
import os
import random
import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt
from gluonts.dataset.pandas import PandasDataset
from sklearn.preprocessing import StandardScaler

class GammaProcessComponent():
    """
    Component for a gamma process.
    """
    def __init__(self, k, lam, initial_state=0.0, num_components=1):
        """
        Args:
            k: shape parameter
            lam: scale parameter
            initial_state: initial state of the component upon reset
            num_components: number of components to simulate
        """
        super().__init__()
        self.k = k
        self.lam = lam
        self.initial_state = initial_state
        self.num_components = num_components

    def reset(self):
        self.state = np.full(self.num_components, self.initial_state)
        return self.state

    def step(self):
        #print("self.k: ", self.k)
        new_gammas = np.random.gamma(self.k, 1 / self.lam, self.num_components)
        #print(f"New gammas: {new_gammas}")
        self.state += new_gammas
        #print(f"New states: {self.state}")
        return self.state

    def get_state(self):
        return self.state
    
    def failed(self) -> bool:
        return False

    def step_size_probability(self, step_size):
        return 1.0 - gamma(a=self.k, scale=1/self.lam).cdf(step_size)

class DeteriorationModel(GammaProcessComponent):
    def __init__(self, lam, b, delta_t, sigma_squared, mu, initial_state=0, num_components=1):
        self.b = b
        self.delta_t = delta_t
        self.sigma_squared = sigma_squared
        self.mu = mu
        self.sigma = np.sqrt(sigma_squared)
        self.num_components = num_components
        super().__init__(k=0, lam=lam, initial_state=initial_state, num_components=num_components)
        self.reset()

    def update_parameters(self):
        m_t = (self.t + self.delta_t)**self.b
        m_t_1 = self.t ** self.b
        self.k = (m_t - m_t_1) * self.a * self.lam

    def step(self):
        self.t += self.delta_t
        self.update_parameters()
        return super().step()
    
    def reset(self):
        self.t = 0
        self.a = np.random.lognormal(self.mu, self.sigma, self.num_components)  # There should be unique a for each component.
        print(f"Initial a: {self.a}")
        return super().reset()

def generate_unique_filename(base_name, extension):
    counter = 1
    base_path = f"datasets/deterioration/{base_name}"
    filename = f"{base_path}.{extension}"
    while os.path.exists(filename):
        filename = f"{base_path}_{counter}.{extension}"
        counter += 1
    return filename


def generate_time_series(params, num_components=5, d_crit=25):
    # Extract parameters from the dictionary
    lam = params['lam']
    b = params['b']
    delta_t = params['delta_t']
    sigma_squared = params['sigma_squared']
    mu = params['mu']
    initial_state = params['initial_state']

    # Initialize the deterioration model
    hierarchical_gamma_process = DeteriorationModel(lam, b, delta_t, sigma_squared, mu, initial_state, num_components)
    #hierarchical_gamma_process.reset()

    # Initialize a list to store the time series for each component
    hierarchical_gamma_series = [[] for _ in range(num_components)]

    # First loop: Generate time series until each component reaches or exceeds d_crit
    for _ in range(int(params['t'] / delta_t)): 
        current_states = hierarchical_gamma_process.step()

        for i in range(num_components):
            if len(hierarchical_gamma_series[i]) == 0 or hierarchical_gamma_series[i][-1] < d_crit:
                hierarchical_gamma_series[i].append(current_states[i])
                if current_states[i] >= d_crit:
                    print(f"Component {i} reached d_crit with value {current_states[i]}")
                    # Stop further generation for this component after recording the first value that exceeds d_crit
                    continue

    # Second loop: Find the maximum length of the generated series
    max_length = max(len(series) for series in hierarchical_gamma_series)

    # Pad the series with NaN at the beginning to match the maximum length
    for i in range(num_components):
        if len(hierarchical_gamma_series[i]) < max_length:
            pad_length = max_length - len(hierarchical_gamma_series[i])
            hierarchical_gamma_series[i] = [np.nan] * pad_length + hierarchical_gamma_series[i]

    # Create a DataFrame for each component's states (wide format)
    data_wide = {}
    for i in range(num_components):
        data_wide[f"Component_{i}"] = hierarchical_gamma_series[i]

    df_wide = pd.DataFrame(data_wide)
    # Add a timestamp index to the DataFrame
    time_index = pd.date_range(start='2021-01-01', periods=max_length, freq=f'{delta_t}H')
    df_wide.index = time_index


    # Generate unique filenames based on parameters
    base_name_wide = f"ts_wide_{params['lam']}_{params['t']}_{num_components}"
    file_name_wide = generate_unique_filename(base_name_wide, "parquet")
    base_name_long = f"ts_long_{params['lam']}_{params['t']}_{num_components}"
    file_name_long = generate_unique_filename(base_name_long, "parquet")

    # Save the wide DataFrame to a Parquet file
    df_wide.to_parquet(file_name_wide)

    df_long = df_wide.reset_index().melt(id_vars=['index'], var_name='item_id', value_name='target')
    df_long.set_index('index', inplace=True)
    df_long.index.name = None

    # Save the long DataFrame to a Parquet file
    df_long.to_parquet(file_name_long)

    return file_name_long, file_name_wide

if __name__ == "__main__":
    # Define parameters as a dictionary
    params = {
        'lam': 1.0,  # Scale parameter
        'b': 2,
        'delta_t': 0.1,
        'sigma_squared': 0.001,  # Standard deviation of the lognormal distribution
        'mu': 0.002,  # Mean of the lognormal distribution
        'initial_state': 0.0,  # Initial state
        't': 100  # Total time
    }

    # Generate the time series and save to Parquet files
    file_name_long, file_name_wide = generate_time_series(params, num_components=10000, d_crit=25)

    print(f"Data saved to {file_name_long} and {file_name_wide}")

    # Function to convert Parquet to CSV
    def convert_parquet_to_csv(parquet_file, csv_file):
        df = pd.read_parquet(parquet_file)
        df.to_csv(csv_file, index=False)

    # Example usage to convert files
    convert_parquet_to_csv(file_name_long, file_name_long.replace(".parquet", ".csv"))
    convert_parquet_to_csv(file_name_wide, file_name_wide.replace(".parquet", ".csv"))