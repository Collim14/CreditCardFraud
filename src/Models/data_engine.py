import polars as pl
import json
import os
from schemas import Features

class DataManager:
    def __init__(self, schema: Features):
        self.schema = schema
        self.df = None
        self.cat_features = []
        self.num_features = []
        self.target_col = None
        self.time_col = None
    
    def process(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File at {data_path} not found")
      
        ext = os.path.splitext(data_path)[1]
        
        if ext == '.csv':
            self.df = pl.scan_csv(data_path)
        elif ext == '.parquet':
            self.df = pl.scan_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        dataset_cols = self.df.columns
        if self.schema.target_col not in dataset_cols:
            raise ValueError(f"Target '{self.schema.target_col}' missing from file")
        reserved_cols = {self.schema.target_col}
        if self.schema.time_col: reserved_cols.add(self.schema.time_col)
        if self.schema.ignore_columns: reserved_cols.update(self.schema.ignore_columns)


        transformations = []
        for col, order_list in self.schema.ordinal_features.items():
            if col in dataset_cols:
                mapping = {val: i for i, val in enumerate(order_list)}

                expr = pl.col(col).replace(mapping, default=None).cast(pl.Float32)
                
                transformations.append(expr)
                
                self.final_num_features.append(col)
                reserved_cols.add(col)
        casts = {}
        for col in self.schema.categorical_features:
            if col in dataset_cols and col not in reserved_cols:
                casts[col] = pl.Categorical
                self.final_cat_features.append(col)
                reserved_cols.add(col)

        for col in self.schema.numerical_features:
            if col in dataset_cols and col not in reserved_cols:
                casts[col] = pl.Float32
                self.final_num_features.append(col)
                reserved_cols.add(col)

        schema = self.df.collect_schema()
        for col, dtype in schema.items():
            if col in reserved_cols:
                continue

            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                         pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                         pl.Float32, pl.Float64]:
                casts[col] = pl.Float32
                self.final_num_features.append(col)

            elif dtype in [pl.String, pl.Boolean, pl.Categorical, pl.Object]:
                casts[col] = pl.Categorical
                self.final_cat_features.append(col)

        #self.df = self.df.with_columns([pl.col(c).cast(t).to_physical() for c,t in casts.items()])

        if transformations:
            self.df = self.df.with_columns(transformations)
        
        if casts:
            self.df = self.df.with_columns([
                pl.col(c).cast(t) for c, t in casts.items()
            ])
            
        if self.schema.ignore_columns:
            valid_drops = [c for c in self.schema.ignore_columns if c in dataset_cols]
            self.df = self.df.drop(valid_drops)
        print("Dataframe processed")
        return self
        
        
    def get_training_data(self):
        if not self.df:
            raise ValueError("Data has not been uploaded")
        return self.df.collect()
    def get_schema_dict(self):
        return self.schema
    def get_schema(self):
       
        return {
            "categorical_features": self.cat_features,
            "numerical_features": self.num_features,
            "target": self.schema.target_col
        }

    def get_hpo_sample(self,time_col, frac = 0.1):
        total = self.df.select(pl.len()).collect().item()
        sample = int(frac*total)
        return self.df.sort(time_col).tail(sample).collect()
    def get_hpo_sparse(self,time_col, frac = 0.1):
        sample = int(1/frac)
        return self.df.sort(time_col).gather_every(sample).collect()