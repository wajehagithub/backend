import sqlite3
from datetime import datetime

def setup_database(db_path: str = "users.db"):
    """Initialize database with required tables"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # User accounts table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        plan_type TEXT DEFAULT 'Free',
        balance REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP
    )
    """)
    
    # Conversation history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        query TEXT NOT NULL,
        response TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        tokens_used INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES user_accounts(id)
    )
    """)
    
    # Knowledge base metadata table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS knowledge_base_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT UNIQUE NOT NULL,
        title TEXT,
        source TEXT,
        category TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata JSON
    )
    """)
    
    # Settings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        setting_key TEXT NOT NULL,
        setting_value TEXT,
        FOREIGN KEY (user_id) REFERENCES user_accounts(id),
        UNIQUE(user_id, setting_key)
    )
    """)
    
    # Insert sample data
    sample_users = [
        ('Alice Smith', 'alice@example.com', 'Premium', 150.00),
        ('Bob Jones', 'bob@example.com', 'Free', 0.00),
        ('Charlie Brown', 'charlie@example.com', 'Pro', 45.50)
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO user_accounts (name, email, plan_type, balance) VALUES (?, ?, ?, ?)",
        sample_users
    )
    
    conn.commit()
    conn.close()
    print(f"✅ Database '{db_path}' initialized successfully!")
    print("📊 Tables created: user_accounts, conversations, knowledge_base_metadata, settings")
    print("👥 Sample users inserted: Alice, Bob, Charlie")