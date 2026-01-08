import sqlite3

DB_PATH = "sentiment.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            text TEXT,
            sentiment TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

def insert_record(timestamp, text, sentiment, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO sentiment_logs (timestamp, text, sentiment, confidence)
        VALUES (?, ?, ?, ?)
    """, (timestamp, text, sentiment, confidence))
    conn.commit()
    conn.close()

def fetch_all():
    conn = sqlite3.connect(DB_PATH)
    df = conn.execute("SELECT * FROM sentiment_logs ORDER BY id DESC").fetchall()
    conn.close()
    return df
