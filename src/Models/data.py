import pandas as pd
import polars as pl
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn import set_config
import os


set_config(transform_output="pandas")

class DataHandler:
    def __init__(self):
        self.df = None
        self.cat_features = []
        self.num_features = []
        self.target_col = None
    
    def process(self, data_path, target_col):
        self.target_col = target_col
        ext = os.path.splitext(data_path)[1]
        
        if ext == '.csv':
            self.df = pl.scan_csv(data_path)
        elif ext == '.parquet':
            self.df = pl.scan_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        schema = self.df.collect_schema()
        casts = {}
        for col,dtype in schema.items():
            if col == self.target_col: continue
            if dtype ==pl.String or dtype == pl.Object:
                casts[col] = pl.Categorical
                self.cat_features.append(col)
            elif dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
                self.num_features.append(col)
       
        self.df = self.df.with_columns([pl.col(c).cast(t).to_physical() for c,t in casts.items()])
        print("Dataframe processed")

    def get_hpo_sample(self,time_col, frac = 0.1):
        total = self.df.select(pl.len()).collect().item()
        sample = int(frac*total)
        return self.df.sort(time_col).tail(sample).collect()
    def get_hpo_sparse(self,time_col, frac = 0.1):
        sample = int(1/frac)
        return self.df.sort(time_col).gather_every(sample).collect()
    

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Iterates through all columns and modifies data type to reduce memory."""
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Mem. usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df
