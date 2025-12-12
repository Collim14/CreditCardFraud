import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn import set_config
import os


set_config(transform_output="pandas")

class DataHandler:
    def __init__(self):
        self.X = None
        self.y = None
    @staticmethod
    def get_preprocessor(X: pd.DataFrame):
   
        numeric_cols = X.select_dtypes(include=['int8', 'int64','float32', 'float64']).columns.tolist()
        
        categorical_cols = X.select_dtypes(include=['category']).columns.tolist()

    
        cat_transformer = OneHotEncoder(handle_unknown='ignore', max_categories=12, min_frequency=3, sparse_output=False, dtype=np.float32)

        preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, categorical_cols)
        ],
        remainder='passthrough'
        )
    
        return preprocessor
    def process(self, data_path, target_col, transform = True):
        self.X, self.y = self.load_data(data_path, target_col)
        obj_cols = self.X.select_dtypes(include = ['object']).columns.tolist()
        str_cols = self.X.select_dtypes(include = ['string']).columns.tolist()
        self.X[obj_cols] = self.X[obj_cols].fillna("")
        self.X[obj_cols] = self.X[obj_cols].astype('category')
        self.X[str_cols] = self.X[str_cols].fillna("")
        self.X[str_cols] = self.X[str_cols].astype('category')

        categorical_cols = self.X.select_dtypes(include=['category']).columns.tolist()
        print(categorical_cols)
        
        
        
        if transform:
            pre = DataHandler.get_preprocessor(self.X)
        
            self.X = pre.fit_transform(self.X) 
        
    
        
        

    @staticmethod
    def load_data(filepath: str, target_col: str):
        ext = os.path.splitext(filepath)[1]
        
        if ext == '.csv':
            df = pd.read_csv(filepath, engine="pyarrow", dtype_backend="pyarrow")
        elif ext == '.parquet':
            df = pd.read_parquet(filepath, dtype_backend="pyarrow")
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        y = df.pop(target_col)
        
        return df, y
    

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
