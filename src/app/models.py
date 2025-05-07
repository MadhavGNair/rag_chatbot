import os

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# define default values
PDF_PATH = "rag_chatbot/pdfs/quaternions.pdf"
MODEL_NAME = "gpt-4o-mini"
API_KEY = os.getenv("OPENAI_API_KEY")
PARENT_MODEL = "openai"


class PDFChatbotInitParams(BaseModel):
    pdf_path: str = PDF_PATH
    model_name: str = MODEL_NAME
    api_key: str = API_KEY
    parent_model: str = PARENT_MODEL


class QueryInput(BaseModel):
    query_string: str
