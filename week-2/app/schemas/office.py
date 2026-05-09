from pydantic import BaseModel
from typing import Optional, Annotated

class office(BaseModel):
    
    officeCode: str

    model_config = {"from_attributes": True}