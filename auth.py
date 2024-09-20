import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = "users.db"

def create_user_table():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        """)
        conn.commit()

def add_user(name, username, password):
    hashed_password = generate_password_hash(password)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        try:
            c.execute("""
                INSERT INTO users (name, username, password)
                VALUES (?, ?, ?)
            """, (name, username, hashed_password))
            conn.commit()
            return None
        except sqlite3.IntegrityError:
            return "Username already exists"

def authenticate_user(username, password):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT * FROM users WHERE username = ?
        """, (username,))
        user = c.fetchone()
        if user and check_password_hash(user[3], password):
            return user
        else:
            return None

def get_user_by_username(username):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            SELECT * FROM users WHERE username = ?
        """, (username,))
        return c.fetchone()
