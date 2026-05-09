from pydantic import BaseModel
from typing import Optional, Annotated

class productLine(BaseModel):
    
    productLine: str

    model_config = {"from_attributes": True}