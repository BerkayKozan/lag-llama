import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, file_path, scaler):
        self.file_path = file_path
        self.df = pd.read_parquet(file_path)
        self.scaler = scaler
    
    def preprocess_data(self):
        for col in self.df.columns:
            if self.df[col].dtype != 'object' and pd.api.types.is_string_dtype(self.df[col]) == False:
                self.df[col] = self.df[col].astype('float32')
    
    def split_data(self):
        total_size = len(self.df)
        train_size = int(total_size * 0.6)
        val_size = int(total_size * 0.2)
        train_df = self.df.iloc[:train_size]
        val_df = self.df.iloc[train_size:train_size + val_size]
        test_df = self.df.iloc[train_size + val_size:]
        return train_df, val_df, test_df

    def normalize_data(self, train_df, val_df, test_df):
        numerical_columns = train_df.select_dtypes(include=['float32', 'float64']).columns
        train_df.loc[:, numerical_columns] = self.scaler.fit_transform(train_df[numerical_columns])
        val_df.loc[:, numerical_columns] = self.scaler.transform(val_df[numerical_columns])
        test_df.loc[:, numerical_columns] = self.scaler.transform(test_df[numerical_columns])
        return train_df, val_df, test_df, self.scaler

    def normalize_with_scaler(self, df, scaler):
        """
        Normalize a dataframe using a preloaded scaler.
        """
        numerical_columns = df.select_dtypes(include=['float32', 'float64']).columns
        df.loc[:, numerical_columns] = scaler.transform(df[numerical_columns])
        return df