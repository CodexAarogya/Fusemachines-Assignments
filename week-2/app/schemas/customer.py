from pydantic import BaseModel
from typing import Optional

class customer_serializer(BaseModel):
    
    customerNumber: int
    customerName: str
    contactLastName: str
    contactFirstName: str
    phone: str
    addressLine1: str
    addressLine2: Optional[str] = None
    city: str
    state: Optional[str] = None
    postalCode: Optional[str] = None
    country: str
    salesRepEmployeeNumber: Optional[int] = None
    creditLimit: float

    model_config = {"from_attributes": True}


class PaginatedCustomers(BaseModel):
    total: int
    page: int
    page_size: int 
    data: list[customer_serializer]

class updateCustomer(BaseModel):
    
    customerName: Optional[str] = None
    contactLastName: Optional[str] = None
    contactFirstName: Optional[str] = None
    phone: Optional[str] = None
    addressLine1: Optional[str] = None
    addressLine2: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postalCode: Optional[str] = None
    country: Optional[str] = None
    salesRepEmployeeNumber: Optional[int] = None
    creditLimit: Optional[float] = None

    model_config = {"from_attributes": True}