from pydantic import BaseModel, Field,  model_validator, validator
from typing import List, Optional, Literal, Dict, Union

class Features(BaseModel):
    target_col: str
    time_col : Optional[str] = None

    categorical_features: List[str] = Field(default_factory=list)
    numerical_features: List[str] = Field(default_factory=list)
    ordinal_features: Dict[str, List[Union[str, int, float]]] = Field(default_factory=dict)
    ignore_columns: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def no_overlap(self):
        cat = set(self.categorical_features)
        num = set(self.numerical_features)
        
        intersection = cat.intersection(num)
        
        if intersection:
            raise ValueError(f"Columns cannot be both categorical and numerical: {intersection}")
            
        return self
    
    @validator('ordinal_features', check_fields=False)
    def validate_ordinal(cls, v):
        for col, order_list in v.items():
            if not order_list:
                raise ValueError(f"Ordinal list for {col} cannot be empty")
        return v
    
class PredictionRequest(BaseModel):
    features: dict