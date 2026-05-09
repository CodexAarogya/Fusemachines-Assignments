from app.database.base import Base
from sqlalchemy.orm import Mapped, mapped_column

class Orders(Base):
    __tablename__="orders"

    orderNumber: Mapped[int] = mapped_column(primary_key=True)