import sys
from pathlib import Path
import uvicorn
from fastapi import FastAPI

# add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.api import initialize_chatbot, query_chatbot

app = FastAPI()


app.post("/init")(initialize_chatbot)
app.post("/query")(query_chatbot)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)