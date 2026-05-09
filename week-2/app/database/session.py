from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings
from dotenv import load_dotenv

load_env = load_dotenv()

engine = create_async_engine(
    url= settings.DATABASE_URL,
    echo = True
)

LocalAsyncSession = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db_connection():
    async with LocalAsyncSession() as LocalSession:
        yield LocalSession
