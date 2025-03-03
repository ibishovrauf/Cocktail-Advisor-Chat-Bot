from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

from app.data_processing import init_vector_db, process_cocktail_data
from app.memory import MemoryManager
from app.rag_engine import RAGEngine


load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

documents, cocktails_df = process_cocktail_data()
vectorstore = init_vector_db(documents)
memory_manager = MemoryManager(vectorstore)
rag_engine = RAGEngine(vectorstore, memory_manager)


@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/api/chat")
async def chat_endpoint(message: str = Form(...)):
    response = rag_engine.query(message)
    return {"response": response}

@app.get("/api/preferences")
async def get_user_preferences():
    preferences = memory_manager.get_preferences()
    return {
        "preferences": [
            {"content": doc.page_content, "type": doc.metadata.get("type", "unknown")}
            for doc in preferences
        ]
    }
