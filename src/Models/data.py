import pandas as pd
import numpy as np

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

def load_data(filepath: str, target_col: str):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        df = reduce_mem_usage(df)
    
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
    elif filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
        df = reduce_mem_usage(df)
    
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
    else:
        raise ValueError("Only CSV and Parquet are supported currently")
        
    
    return X, y