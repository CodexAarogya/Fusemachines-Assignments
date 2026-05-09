from pydantic import BaseModel
from typing import Optional, Annotated

class order(BaseModel):
    
    orderNumber: int

    model_config = {"from_attributes": True}