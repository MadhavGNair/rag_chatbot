import sys
from pathlib import Path
from fastapi import HTTPException

# add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.models import PDFChatbotInitParams, QueryInput
from rag_chatbot.pdf_chatbot import PDFChatbot

# define the global chatbot instance
chatbot: "PDFChatbot | None" = None


async def initialize_chatbot(init_params: PDFChatbotInitParams):
  """Initializes the PDFChatbot with the provided parameters.
  If no parameters are provided, uses the default values.
  """
  global chatbot
  try:
      chatbot = PDFChatbot(
          pdf_path=init_params.pdf_path,
          model_name=init_params.model_name,
          api_key=init_params.api_key,
          parent_model=init_params.parent_model
      )
      return {"message": "Chatbot initialized successfully"}
  except Exception as e:
      raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


async def query_chatbot(query_input: QueryInput):
  """Queries the PDFChatbot and returns the answer.
  If the chatbot is not initialized, it will be initialized with default values.
  """
  global chatbot

  if chatbot is None:
      # initialize with default values if not already initialized
      try:
          default_params = PDFChatbotInitParams()
          chatbot = PDFChatbot(
              pdf_path=default_params.pdf_path,
              model_name=default_params.model_name,
              api_key=default_params.api_key,
              parent_model=default_params.parent_model
          )
          print("Chatbot initialized with default values.")
      except Exception as e:
          raise HTTPException(
              status_code=500,
              detail=f"Failed to initialize with default values: {str(e)}",
          )

  try:
      answer = chatbot.query(query=query_input.query_string)
      return {"answer": answer}
  except Exception as e:
      raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")