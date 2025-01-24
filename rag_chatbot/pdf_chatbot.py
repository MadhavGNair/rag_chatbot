from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class PDFChatbot:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdf_name = os.path.basename(pdf_path)

    def __load_and_split(self):
        # load the text from the pdf
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load_and_split()
        text = "\n".join([doc.page_content for doc in pages])

        # split the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )
        docs = text_splitter.create_documents([text])
        for i, d in enumerate(docs):
            d.metadata = {"doc_id": i}
        return docs

    def __generate_embeddings(self, docs):
        if os.path.exists(f'./embeddings/{self.pdf_name}.csv'):
            print(f'Loading embeddings from file: {self.pdf_name}.csv')
            return pd.read_csv(f'./embeddings/{self.pdf_name}.csv')
        
        # If file doesn't exist, generate new embeddings
        print(f'Generating embeddings for {self.pdf_name}')
        def get_embeddings(text):
            model = 'models/text-embedding-004'
            embedding = genai.embed_content(model=model,
                                            content=text,
                                            task_type="retrieval_document")
            return embedding['embedding']
        
        content_list = [doc.page_content for doc in docs]
        embeddings = [get_embeddings(content) for content in content_list]

        dataframe = pd.DataFrame({
            'page_content': content_list,
            'embeddings': embeddings
        })
        dataframe.to_csv(f'./embeddings/{self.pdf_name}.csv', index=False)
        return dataframe

    def query_chatbot(self):
        docs = self.__load_and_split()
        df = self.__generate_embeddings(docs)
        print(df.head())


if __name__ == "__main__":
    pdf_path = "./pdfs/quaternions.pdf"
    chatbot = PDFChatbot(pdf_path)
    chatbot.query_chatbot()