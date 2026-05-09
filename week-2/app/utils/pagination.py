from fastapi import Query

class Pagination :
    def __init__(self, page: int = Query(default=1, ge=1), page_size: int = Query(default=10, ge=1, le=30)):
        self.page = page
        self.page_size = page_size

    @property
    def offset(self):
        return (self.page - 1) * self.page_size
