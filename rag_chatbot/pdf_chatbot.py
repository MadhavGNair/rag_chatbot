import os
import logging
from dotenv import load_dotenv

from google import genai

from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
logging.basicConfig(level=logging.INFO)


class PDFChatbot:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.model_name = 'gemini-2.0-flash-exp'

        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options={"api_version": "v1alpha"},
        )

    def load_pdf(self):
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        texts = [doc.page_content for doc in docs]
        return texts

    def summarize_texts(self, texts):
        prompt = """You are an expert tasked with summarizing text for retrieval. \
                    These summaries will be embedded and used to retrieve the raw text. \
                    Give a concise summary of the text that is well optimized for retrieval. Text: {element} """
        summaries = []
        for text in texts:
            response = self.client.models.generate_content(
                model=self.model_name, contents=[text, prompt]
            )
            summaries.extend(response.text)

        with open('summaries.txt', 'w') as f:
            f.write('\n===\n'.join(summaries))


if __name__ == "__main__":
    pdf_path = "./pdfs/quaternions.pdf"
    chatbot = PDFChatbot(pdf_path)
    texts = chatbot.load_pdf()
    chatbot.summarize_texts(texts)

