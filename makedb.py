import sqlite3
from werkzeug.security import generate_password_hash

# Connect to SQLite database
conn = sqlite3.connect('users.db')

# Create a cursor
cursor = conn.cursor()

# Create the users table (if it doesn't exist already)
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')

# Hash the password before inserting
email = 'user@example.com'
password = 'yourpassword'  # The plain text password you want to store

# Hash the password using pbkdf2:sha256
hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

# Insert a new user with the hashed password
cursor.execute('''
INSERT INTO users (email, password)
VALUES (?, ?)
''', (email, hashed_password))

# Commit and close connection
conn.commit()
conn.close()

print("User added successfully with a hashed password!")
