import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import HuggingFaceHub
import pinecone
from pinecone import ServerlessSpec


class PDFChatbot:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def __parse_pdf(self):
        loader = PyMuPDFLoader(self.pdf_path)
        docs = loader.load()
        return docs

    def __initialize_vector_space(self, docs):
        pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index_name = 'pdf_chatbot'
        embeddings = HuggingFaceEmbeddings()

        if index_name not in pc.list_indexes():
            pc.create_index(name=index_name, metric='cosine', dimension=768, spec=ServerlessSpec(cloud='gcp', region='europe-west4'))
            docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        else:
            docsearch = Pinecone.from_existing_index(index_name, embeddings)

    def __initialize_llm(self):
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
        )
        return llm


    def chat(self):
        documents = self.__parse_pdf()


if __name__ == '__main__':
    chatbot = PDFChatbot('./pdfs/quaternions.pdf')
    chatbot.chat()
