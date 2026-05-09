from app.database.base import Base
from sqlalchemy.orm import Mapped, mapped_column

class Employee(Base):
    __tablename__="employees"

    employeeNumber: Mapped[int] = mapped_column(primary_key=True)