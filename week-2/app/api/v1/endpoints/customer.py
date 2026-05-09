from fastapi import APIRouter, Depends, Path , Query ,HTTPException
from app.models.customer_model import Customer
from app.schemas.customer import PaginatedCustomers, customer_serializer, updateCustomer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database.session import get_db_connection
from app.utils.paginationType import PaginationType

router = APIRouter(
    prefix="/customers",
    tags=["Customers"]
)


@router.get("/", response_model=PaginatedCustomers)
async def fetch_all_customers(
    pagination: PaginationType,
    db: AsyncSession = Depends(get_db_connection),
    ):

    total_results = await db.execute(select(func.count()).select_from(Customer))
    total_results = total_results.scalar()

    results = await db.execute(
        select(Customer)
        .offset(pagination.offset)
        .limit(pagination.page_size)
    )
    
    customers = results.scalars().all()

    return PaginatedCustomers(
        total=total_results,
        page=pagination.page,
        page_size=pagination.page_size,
        data=customers
    )


@router.get("/customer")
async def fetch_customer_by_customerNumber(
    db: AsyncSession = Depends(get_db_connection),
    customerNumber: int = Query(ge=1)
    ):

    result = await db.execute(
        select(Customer)
        .where(Customer.customerNumber == customerNumber))
    
    customer = result.scalar_one_or_none()

    if customer is None:
        raise HTTPException(
            status_code=404,
            detail="Customer not found!"
        )
    
    else:
        return customer
    
    
@router.delete("/delete")
async def delete_customer_by_customerNumber(
    customerNumber: int = Query(ge=1),
    db: AsyncSession = Depends(get_db_connection)
    ):

    result = await db.execute(
        select(Customer)
        .where(Customer.customerNumber == customerNumber)
    )

    customer = result.scalar_one_or_none()

    if customer is None:
        raise HTTPException(
            status_code=404,
            detail="Customer not found!",
        )
    else:
        await db.delete(customer)
        await db.commit()
        return(
            {"Message": f"customer with customer no. {customerNumber} has been deleted successfully!"}
        )


@router.patch("/customer/{customerNumber}")
async def patch_customer_by_customerNumber(customerData: updateCustomer,
                                            customerNumber: int = Path(ge=1),
                                            db: AsyncSession = Depends(get_db_connection),
                                      ):
                                      

    result = await db.execute(
        select(Customer)
        .where(Customer.customerNumber == customerNumber)
    )

    customer = result.scalar_one_or_none()
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found!")
    
    else:
        update_data = customerData.model_dump(exclude_unset=True)

        for key, value in update_data.items():
            setattr(customer, key, value)

        await db.commit()
        await db.refresh(customer)

        return ({"Message": "Customer data updated successfully!"})
    


@router.put("/customer/{customerNumber}")
async def put_customer_by_customerNumber(customerData: customer_serializer,
                                            customerNumber: int = Path(ge=1),
                                            db: AsyncSession = Depends(get_db_connection),
                                      ):
                                      

    result = await db.execute(
        select(Customer)
        .where(Customer.customerNumber == customerNumber)
    )

    customer = result.scalar_one_or_none()
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found!")
    
    else:
        update_data = customerData.model_dump(exclude_unset=False)

        for key, value in update_data.items():
            setattr(customer, key, value)

        await db.commit()
        await db.refresh(customer)

        return ({"Message": "Customer data updated successfully!"})
