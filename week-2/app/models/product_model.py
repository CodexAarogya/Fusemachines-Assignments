from app.database.base import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String

class Product(Base):
    __tablename__="products"

    productCode: Mapped[str] = mapped_column(String(15) ,primary_key=True)