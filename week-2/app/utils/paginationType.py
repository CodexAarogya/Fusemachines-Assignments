from typing import Annotated
from app.utils.pagination import Pagination
from fastapi import Depends

PaginationType = Annotated[Pagination, Depends(Pagination)]