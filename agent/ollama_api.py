from fastapi import FastAPI, HTTPException
from ollama import Client
from fastapi import Body
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

app = FastAPI(
    title="Ollama Chat API",
    description="A simple API for chatting with Ollama models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client(
    host="http://localhost:11434"
)
client.pull("gemma3:1b")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Ollama Chat API",
        "endpoints": {
            "/chat": "POST - Send messages to chat with the AI",
            "/docs": "GET - View API documentation"
        }
    }

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    try:
        if not chat_request.messages:
            return JSONResponse(
                status_code=400,
                content={"error": "Messages array cannot be empty"}
            )
        
        response = client.chat(
            model="gemma3:1b",
            messages=[{"role": m.role, "content": m.content} for m in chat_request.messages]
        )
        return {"response": response['message']['content']}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )

@app.get("/example")
def get_example():
    return {
        "example_request": {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?"
                }
            ]
        }
    }
