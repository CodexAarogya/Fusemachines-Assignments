from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database.session import get_db_connection
from app.models import (
    customer_model,
    order_model,
    employee_model,
    office,
    product_model,
    productline_model,
)
from app.schemas.count_response import CountResponse

router = APIRouter(
    prefix="/count",
    tags=["Count_all"],
)


@router.get("/", response_model=CountResponse)
async def get_counts(db: AsyncSession = Depends(get_db_connection)):

    customers = await db.scalar(
        select(func.count(customer_model.Customer.customerNumber))
    )

    orders = await db.scalar(
        select(func.count(order_model.Orders.orderNumber))
    )

    employees = await db.scalar(
        select(func.count(employee_model.Employee.employeeNumber))
    )

    offices = await db.scalar(
        select(func.count(office.Office.officeCode))
    )

    products = await db.scalar(
        select(func.count(product_model.Product.productCode))
    )

    productlines = await db.scalar(
        select(func.count(productline_model.ProductLine.productLine))
    )

    return CountResponse(
        customers=customers,
        orders=orders,
        employees=employees,
        offices=offices,
        products=products,
        productlines=productlines,
    )