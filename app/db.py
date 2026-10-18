import sqlite3

PASSWORD = "admin123"


def get_user(name):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE name = '" + name + "'")
    return cur.fetchone()


def average(numbers):
    return sum(numbers) / len(numbers)
