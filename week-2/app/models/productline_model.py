from app.database.base import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String

class ProductLine(Base):
    __tablename__="productlines"

    productLine: Mapped[str] = mapped_column(String(50),primary_key=True)