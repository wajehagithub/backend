"""
Database Service Module - Handles both SQLite and Pinecone interactions
"""
import sqlite3
from typing import Optional, List, Dict, Any
import json
from datetime import datetime
from contextlib import contextmanager

class DatabaseService:
    """Service for managing local SQLite database"""
    
    def __init__(self, db_path: str = "users.db"):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
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
            
            conn.commit()
            self._insert_sample_data(conn)
    
    def _insert_sample_data(self, conn):
        """Insert sample data if tables are empty"""
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM user_accounts")
        
        if cursor.fetchone()[0] == 0:
            sample_users = [
                (1, 'Alice Smith', 'alice@example.com', 'Premium', 150.00),
                (2, 'Bob Jones', 'bob@example.com', 'Free', 0.00),
                (3, 'Charlie Brown', 'charlie@example.com', 'Pro', 45.50)
            ]
            cursor.executemany(
                "INSERT OR IGNORE INTO user_accounts (id, name, email, plan_type, balance) VALUES (?,?,?,?,?)",
                sample_users
            )
            conn.commit()
    
    # User Methods
    def get_user(self, user_id: int) -> Optional[Dict]:
        """Retrieve user information"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_accounts WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Retrieve user by email"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_accounts WHERE email = ?", (email,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def create_user(self, name: str, email: str, plan_type: str = "Free") -> Dict:
        """Create new user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_accounts (name, email, plan_type) VALUES (?, ?, ?)",
                (name, email, plan_type)
            )
            conn.commit()
            user_id = cursor.lastrowid
            return self.get_user(user_id)
    
    def update_user(self, user_id: int, **kwargs) -> Dict:
        """Update user information"""
        allowed_fields = {'name', 'email', 'plan_type', 'balance', 'last_login'}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return self.get_user(user_id)
        
        set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [user_id]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"UPDATE user_accounts SET {set_clause} WHERE id = ?", values)
            conn.commit()
        
        return self.get_user(user_id)
    
    # Conversation Methods
    def save_conversation(self, user_id: int, query: str, response: str, tokens_used: int = 0) -> Dict:
        """Save conversation to database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO conversations (user_id, query, response, tokens_used) VALUES (?, ?, ?, ?)",
                (user_id, query, response, tokens_used)
            )
            conn.commit()
            conversation_id = cursor.lastrowid
        
        return self.get_conversation(conversation_id)
    
    def get_conversation(self, conversation_id: int) -> Optional[Dict]:
        """Retrieve conversation"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_user_conversations(self, user_id: int, limit: int = 50) -> List[Dict]:
        """Get user's conversation history"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    # Knowledge Base Methods
    def save_knowledge_metadata(self, document_id: str, title: str, source: str, 
                               category: str, metadata: Dict = None) -> Dict:
        """Save knowledge base document metadata"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                INSERT INTO knowledge_base_metadata 
                (document_id, title, source, category, metadata) 
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET 
                updated_at = CURRENT_TIMESTAMP, 
                metadata = ?
            """, (document_id, title, source, category, metadata_json, metadata_json))
            conn.commit()
        
        return self.get_knowledge_metadata(document_id)
    
    def get_knowledge_metadata(self, document_id: str) -> Optional[Dict]:
        """Retrieve knowledge metadata"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_base_metadata WHERE document_id = ?", (document_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
                return result
        return None
    
    def get_all_knowledge_metadata(self, category: str = None) -> List[Dict]:
        """Get all knowledge metadata, optionally filtered by category"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute(
                    "SELECT * FROM knowledge_base_metadata WHERE category = ? ORDER BY created_at DESC",
                    (category,)
                )
            else:
                cursor.execute("SELECT * FROM knowledge_base_metadata ORDER BY created_at DESC")
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
                results.append(result)
            return results
    
    # Settings Methods
    def save_setting(self, user_id: int, setting_key: str, setting_value: str) -> Dict:
        """Save user setting"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO settings (user_id, setting_key, setting_value)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, setting_key) DO UPDATE SET setting_value = ?
            """, (user_id, setting_key, setting_value, setting_value))
            conn.commit()
        
        return self.get_setting(user_id, setting_key)
    
    def get_setting(self, user_id: int, setting_key: str) -> Optional[str]:
        """Get user setting"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT setting_value FROM settings WHERE user_id = ? AND setting_key = ?",
                (user_id, setting_key)
            )
            row = cursor.fetchone()
            return row[0] if row else None
    
    def get_user_settings(self, user_id: int) -> Dict[str, str]:
        """Get all user settings"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT setting_key, setting_value FROM settings WHERE user_id = ?",
                (user_id,)
            )
            return {row[0]: row[1] for row in cursor.fetchall()}
