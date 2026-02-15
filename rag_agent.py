"""
RAG Agent using LangGraph - Handles multi-database RAG with LangChain
"""
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_core.tools import Tool, create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

from database_service import DatabaseService

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """State definition for RAG agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: int
    context: Dict[str, Any]


class RAGChatbot:
    """RAG Chatbot using LangGraph"""
    
    def __init__(self, db_service: Optional[DatabaseService] = None):
        """
        Initialize RAG Chatbot
        
        Args:
            db_service: DatabaseService instance for local database
        """
        self.db_service = db_service or DatabaseService()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=2048
        )
        
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "rag12")
        self.vectorstore = None
        self.graph = self._build_graph()
    
    def _init_vectorstore(self):
        """Initialize Pinecone vector store (v3 API)"""
        if not self.vectorstore:
            try:
                from pinecone import Pinecone

                pc = Pinecone(
                    api_key=os.getenv("PINECONE_API_KEY")
                )

                index = pc.Index(self.index_name)

                from langchain_community.vectorstores import Pinecone as PineconeStore

                self.vectorstore = PineconeStore(
                    index=index,
                    embedding=self.embeddings,
                    text_key="text"
                )

                logger.info(f"✅ Connected to Pinecone index: {self.index_name}")

            except Exception as e:
                logger.error(f"❌ Failed to connect to Pinecone: {e}")
                raise
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the RAG agent"""
        tools = []
        
        # Pinecone retrieval tool
        self._init_vectorstore()
        pinecone_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        pinecone_tool = create_retriever_tool(
            pinecone_retriever,
            "pinecone_search",
            "Search the knowledge base in Pinecone for relevant information. Use this when you need specific product/company information."
        )
        tools.append(pinecone_tool)
        
        # Local database search tool
        def search_local_database(query: str) -> str:
            """Search local database for user-related information"""
            try:
                metadata_list = self.db_service.get_all_knowledge_metadata()
                
                if not metadata_list:
                    return "No local knowledge base entries found."
                
                # Simple keyword matching for local search
                query_lower = query.lower()
                matches = [
                    m for m in metadata_list
                    if query_lower in m.get('title', '').lower() or
                       query_lower in m.get('category', '').lower() or
                       query_lower in str(m.get('metadata', '')).lower()
                ]
                
                if matches:
                    results = "\n".join([
                        f"- {m['title']} (Category: {m['category']}, Source: {m['source']})"
                        for m in matches[:3]
                    ])
                    return f"Found local knowledge base entries:\n{results}"
                else:
                    return "No matching entries found in local knowledge base."
            except Exception as e:
                logger.error(f"Error searching local database: {e}")
                return f"Error searching local database: {str(e)}"
        
        local_db_tool = Tool(
            name="local_database_search",
            func=search_local_database,
            description="Search the local SQLite database for company-specific, user, or historical information. Use this for account details, settings, or internal data."
        )
        tools.append(local_db_tool)
        
        # User profile tool
        def get_user_profile(user_id: int) -> str:
            """Get user profile information"""
            try:
                user = self.db_service.get_user(user_id)
                if user:
                    return f"User: {user['name']}, Email: {user['email']}, Plan: {user['plan_type']}, Balance: ${user['balance']}"
                else:
                    return f"User {user_id} not found."
            except Exception as e:
                return f"Error fetching user profile: {str(e)}"
        
        user_profile_tool = Tool(
            name="get_user_profile",
            func=get_user_profile,
            description="Get the current user's profile information including name, email, and plan type."
        )
        tools.append(user_profile_tool)
        
        return tools
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(RAGState)
        
        tools = self._create_tools()
        
        # Create agent runnable
        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent(
            self.llm,
            tools,
        )
        
        def process_message(state: RAGState) -> RAGState:
            """Process message through the agent"""
            try:
                # Run the agent
                output = agent.invoke({
                    "messages": state["messages"]
                })
                
                # Extract response
                if "messages" in output:
                    state["messages"] = output["messages"]
                
                return state
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                error_message = AIMessage(content=f"Error processing request: {str(e)}")
                state["messages"].append(error_message)
                return state
        
        def save_conversation(state: RAGState) -> RAGState:
            """Save conversation to database"""
            try:
                if state.get("messages") and len(state["messages"]) >= 2:
                    last_human = None
                    last_ai = None
                    
                    for msg in reversed(state["messages"]):
                        if isinstance(msg, HumanMessage) and not last_human:
                            last_human = msg.content
                        elif isinstance(msg, AIMessage) and not last_ai:
                            last_ai = msg.content
                        
                        if last_human and last_ai:
                            break
                    
                    if last_human and last_ai and state.get("user_id"):
                        self.db_service.save_conversation(
                            user_id=state["user_id"],
                            query=last_human,
                            response=last_ai,
                            tokens_used=0
                        )
            except Exception as e:
                logger.error(f"Error saving conversation: {e}")
            
            return state
        
        # Add nodes
        workflow.add_node("agent", process_message)
        workflow.add_node("save", save_conversation)
        
        # Add edges
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", "save")
        workflow.add_edge("save", END)
        
        return workflow.compile()
    
    async def chat(self, user_id: int, query: str) -> str:
        """
        Process user query through RAG
        
        Args:
            user_id: User ID for context
            query: User query
            
        Returns:
            AI response
        """
        try:
            # Initialize state
            state = RAGState(
                messages=[HumanMessage(content=query)],
                user_id=user_id,
                context={}
            )
            
            # Run graph
            result = self.graph.invoke(state)
            
            # Extract response
            if result.get("messages"):
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        return msg.content
            
            return "No response generated."
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error processing your request: {str(e)}"
    
    def add_document_to_knowledge_base(self, document_id: str, title: str, 
                                       source: str, category: str, 
                                       content: str, metadata: Dict = None) -> bool:
        """
        Add document to local knowledge base
        
        Args:
            document_id: Unique document ID
            title: Document title
            source: Document source
            category: Document category
            content: Document content
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            self.db_service.save_knowledge_metadata(
                document_id=document_id,
                title=title,
                source=source,
                category=category,
                metadata={
                    "content": content,
                    "added_at": datetime.now().isoformat(),
                    **(metadata or {})
                }
            )
            logger.info(f"✅ Added document to knowledge base: {document_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Error adding document: {e}")
            return False


# Singleton instance
_chatbot_instance: Optional[RAGChatbot] = None

def get_chatbot() -> RAGChatbot:
    """Get or create chatbot instance"""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = RAGChatbot()
    return _chatbot_instance