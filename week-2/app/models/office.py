from app.database.base import Base
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String

class Office(Base):
    __tablename__="offices"

    officeCode: Mapped[str] = mapped_column(String(10), primary_key=True)