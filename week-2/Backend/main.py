from fastapi import FastAPI, Query
from typing import Annotated
from dotenv import load_dotenv
import psycopg2, os

app = FastAPI(title="Week-2 Agentic Software Development")
load_env = load_dotenv("../.env")

def get_db_connection():
    conn = psycopg2.connect(
        host="db",
        database = os.getenv("POSTGRES_DB"),
        user = os.getenv("POSTGRES_USER"),
        password = os.getenv("POSTGRES_PASSWORD")
    )
    return conn

@app.get("/")
async def root():
    return {"Message": "This is Home"}

@app.get("/db-check")
async def db_check():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT version();')
        version_info = cursor.fetchone()
        cursor.close()
        conn.close()
        return {"Database version": version_info}
    
    except Exception as e:
        return {"Error": e}
    


