import psycopg2
from psycopg2 import OperationalError
from dotenv import load_dotenv
import os 

# Завантаження змінних середовища з .env
load_dotenv()

# Функція для створення підключення до PostgreSQL
def create_connection():
    try:
        # Параметри підключення з .env
        connection = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        print("✅ Successfully connected to PostgreSQL")
        return connection
    except OperationalError as e:
        print(f"❌ Connection failed: {e}")
        return None
