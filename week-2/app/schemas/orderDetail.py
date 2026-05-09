from pydantic import BaseModel
from typing import Optional, Annotated

class orderDetail(BaseModel):
    
    orderNumber: int
    productCode: str

    model_config = {"from_attributes": True}