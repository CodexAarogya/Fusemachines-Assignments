from pydantic import BaseModel
from app.schemas import customer, employee, office, order, orderDetail, payment, product, productLine

class CountResponse(BaseModel):

    customers: int|None = None
    orders: int|None = None
    products: int|None = None
    employees: int|None = None
    offices: int|None = None
    productlines: int|None = None
    