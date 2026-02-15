"""
RAG Chatbot Backend - FastAPI with LangGraph and Dual Database Support
Production-ready RAG system with Pinecone and SQLite integration
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional
import json

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from database_service import DatabaseService
from rag_agent import get_chatbot

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global dependencies
db_service: Optional[DatabaseService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    global db_service
    
    # Startup
    logger.info("🚀 Starting RAG Chatbot Backend...")
    db_service = DatabaseService()
    chatbot = get_chatbot()  # Initialize chatbot
    logger.info("✅ Backend initialized successfully!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down RAG Chatbot Backend...")


# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="RAG-based chatbot with LangGraph, Pinecone, and SQLite",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", '["http://localhost:3000"]')
try:
    origins_list = json.loads(allowed_origins)
except:
    origins_list = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "*",],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    """Chat request model"""
    query: str
    user_id: int = 1

class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    user_id: int
    timestamp: str

class UserProfile(BaseModel):
    """User profile model"""
    name: str
    email: str
    plan_type: str = "Free"

class KnowledgeDocumentRequest(BaseModel):
    """Request model for adding knowledge document"""
    document_id: str
    title: str
    source: str
    category: str
    content: str

# API Endpoints

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "RAG Chatbot Backend is running",
        "version": "1.0.0"
    }

@app.get("/health", tags=["Health"])
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "embeddings": "initialized",
        "llm": "ready"
    }

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Process user query through RAG chatbot
    
    Request body:
    - query: The user's question or prompt
    - user_id: Optional user ID (default: 1)
    
    Returns:
    - answer: AI response
    - user_id: The user ID used
    - timestamp: When the response was generated
    """
    try:
        from datetime import datetime
        
        chatbot = get_chatbot()
        response = await chatbot.chat(
            user_id=request.user_id,
            query=request.query
        )
        
        return ChatResponse(
            answer=response,
            user_id=request.user_id,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat: {str(e)}"
        )

@app.get("/user/{user_id}", tags=["Users"])
async def get_user_profile(user_id: int):
    """Get user profile information"""
    try:
        if not db_service:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        user = db_service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user", tags=["Users"])
async def create_user(profile: UserProfile):
    """Create a new user"""
    try:
        if not db_service:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        user = db_service.create_user(
            name=profile.name,
            email=profile.email,
            plan_type=profile.plan_type
        )
        return user
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/conversations", tags=["Conversations"])
async def get_conversations(user_id: int, limit: int = 50):
    """Get user's conversation history"""
    try:
        if not db_service:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        conversations = db_service.get_user_conversations(user_id, limit)
        return {"conversations": conversations, "total": len(conversations)}
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/add", tags=["Knowledge Base"])
async def add_knowledge_document(request: KnowledgeDocumentRequest):
    """Add a document to the knowledge base"""
    try:
        chatbot = get_chatbot()
        success = chatbot.add_document_to_knowledge_base(
            document_id=request.document_id,
            title=request.title,
            source=request.source,
            category=request.category,
            content=request.content
        )
        
        if success:
            return {"status": "success", "message": "Document added to knowledge base"}
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to add document to knowledge base"
            )
    except Exception as e:
        logger.error(f"Error adding knowledge document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge/metadata", tags=["Knowledge Base"])
async def get_knowledge_metadata(category: Optional[str] = None):
    """Get knowledge base metadata"""
    try:
        if not db_service:
            raise HTTPException(status_code=500, detail="Database not initialized")
        
        metadata = db_service.get_all_knowledge_metadata(category)
        return {"metadata": metadata, "total": len(metadata)}
    except Exception as e:
        logger.error(f"Error fetching knowledge metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )