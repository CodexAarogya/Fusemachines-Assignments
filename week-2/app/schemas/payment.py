from pydantic import BaseModel
from typing import Optional, Annotated

class payment(BaseModel):
    
    customerNumber: int
    checkNumber: str

    model_config = {"from_attributes": True}