from pydantic import BaseModel
from typing import Optional

class employee(BaseModel):
    
    employeeNumber: int

    model_config = {"from_attributes": True}