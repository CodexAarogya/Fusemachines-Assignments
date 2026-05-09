from pydantic import BaseModel
from typing import Optional, Annotated

class product(BaseModel):
    
    productCode: str

    model_config = {"from_attributes": True}