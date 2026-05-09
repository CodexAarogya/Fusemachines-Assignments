from app.database.base import Base
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import String, ForeignKey, Numeric

class Customer(Base):
    __tablename__= "customers"

    customerNumber: Mapped[int] = mapped_column(primary_key=True)
    customerName: Mapped[str] = mapped_column(String(50), nullable=False)
    contactLastName: Mapped[str] = mapped_column(String(50), nullable=False)
    contactFirstName: Mapped[str] = mapped_column(String(50), nullable=False)
    phone: Mapped[str] = mapped_column(String(50), nullable=False)
    addressLine1: Mapped[str] = mapped_column(String(50), nullable=False)
    addressLine2: Mapped[str] = mapped_column(String(50), nullable=True)
    city: Mapped[str] = mapped_column(String(50), nullable=False)
    state: Mapped[str] = mapped_column(String(50), nullable=True)
    postalCode: Mapped[str] = mapped_column(String(15), nullable=True)
    country: Mapped[str] = mapped_column(String(50), nullable=False)
    salesRepEmployeeNumber: Mapped[int]
    creditLimit: Mapped[float] = mapped_column(Numeric(10,2))

    
    