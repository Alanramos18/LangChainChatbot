from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import uvicorn

load_dotenv()

# Request/Response Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: Optional[List[ChatMessage]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    sources: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    message: str

# Chatbot Service
class EngineeringCompanionService:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        self.faiss_index = FAISS.load_local(
            "faiss_index", 
            embeddings=self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.retriever = self.faiss_index.as_retriever(search_kwargs={"k": 5})
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )
        self.sessions = {}
        self._setup_chain()
    
    def _setup_chain(self):
        """Configure the RAG chain with enhanced system prompt"""
        system_prompt = """Eres un Asistente de Ingeniería Informática, un compañero virtual especializado en apoyar a estudiantes de ingeniería en informática.

TUS CAPACIDADES:

1. **Resúmenes de Temas**: Puedes proporcionar resúmenes claros y concisos de cualquier tema técnico, concepto de programación, algoritmo o materia de la carrera.

2. **Información de Horarios**: Tienes acceso a información sobre horarios de clases, fechas de exámenes, y calendarios académicos. Puedes ayudar a organizar y recordar fechas importantes.

3. **Evaluaciones Rápidas**: Puedes generar preguntas de práctica, cuestionarios rápidos y ayudar a verificar comprensión de temas mediante evaluaciones interactivas.

4. **Soporte Académico**: Respondes preguntas sobre tareas, proyectos, conceptos de programación, estructuras de datos, algoritmos, bases de datos, y cualquier materia de la carrera.

INSTRUCCIONES:
- Sé claro, preciso y educativo en tus respuestas
- Usa ejemplos cuando sea apropiado
- Si no tienes información suficiente en el contexto, indícalo claramente
- Mantén un tono amigable pero profesional
- Cita las fuentes cuando uses información específica del contexto

Usa el siguiente contexto de la documentación del curso para responder la pregunta:

{context}

Responde a la pregunta del estudiante de manera útil y educativa."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        combine_chain = create_stuff_documents_chain(llm=self.llm, prompt=self.prompt)
        self.qa_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=combine_chain
        )
    
    def process_message(self, request: ChatRequest) -> ChatResponse:
        """Process a chat message and return response"""
        try:
            # Generate session ID if not provided
            session_id = request.session_id or f"session_{datetime.now().timestamp()}"
            
            # Build conversation context if history exists
            full_context = request.message
            if request.history:
                history_text = "\n".join([
                    f"{'Usuario' if msg.role == 'user' else 'Asistente'}: {msg.content}"
                    for msg in request.history[-5:]  # Last 5 messages for context
                ])
                full_context = f"Conversación previa:\n{history_text}\n\nPregunta actual: {request.message}"
            
            # Invoke the RAG chain
            result = self.qa_chain.invoke({"input": full_context})
            
            # Extract source documents
            sources = []
            if "context" in result:
                sources = [doc.metadata.get("source", "Unknown") for doc in result["context"]]
                sources = list(set(sources))  # Remove duplicates
            
            return ChatResponse(
                response=result["answer"],
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                sources=sources if sources else None
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

# FastAPI App
app = FastAPI(
    title="Engineering Companion API",
    description="AI-powered chatbot for engineering students",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
companion_service = EngineeringCompanionService()

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="online",
        message="Engineering Companion API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint
    
    Send a message to the Engineering Companion and receive a response
    with relevant information from the course documentation.
    """
    return companion_service.process_message(request)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint (placeholder for future implementation)
    
    This endpoint can be implemented for streaming responses.
    """
    raise HTTPException(status_code=501, detail="Streaming not yet implemented")

if __name__ == "__main__":
    uvicorn.run(
        "service:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
