import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def get_preprocessor(X: pd.DataFrame):
    """
    Creates a scikit ColumnTransformer based on input dataframe types.
    """
    numeric_cols = X.select_dtypes(include=['int8', 'int64','float32', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    num_transformer = StandardScaler()
    
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_cols),
            ('cat', cat_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor