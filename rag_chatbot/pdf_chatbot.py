import os

import clickhouse_connect
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_ID = "gemini-2.0-flash-exp"


class PDFChatbot:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdf_name = os.path.basename(pdf_path)
        self.db_client = clickhouse_connect.get_client(
            host="msc-8f512545.us-east-1.aws.myscale.com",
            port=443,
            username="madhav_gn_org_default",
            password="passwd_uuTIJIM478xafg",
        )

    def __load_and_split(self):
        print(f"Loading and splitting text from {self.pdf_name}...")
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

    def __get_embeddings(self, text):
        model = "models/text-embedding-004"
        embedding = genai.embed_content(
            model=model, content=text, task_type="retrieval_document"
        )
        return embedding["embedding"]

    def __generate_embeddings(self, docs):
        if os.path.exists(f"./embeddings/{self.pdf_name}.csv"):
            print(f"Loading embeddings from file: {self.pdf_name}.csv...")
            return pd.read_csv(f"./embeddings/{self.pdf_name}.csv")

        # if file doesn't exist, generate new embeddings
        print(f"Generating embeddings for {self.pdf_name}...")

        content_list = [doc.page_content for doc in docs]
        embeddings = [self.__get_embeddings(content) for content in content_list]

        dataframe = pd.DataFrame(
            {"page_content": content_list, "embeddings": embeddings}
        )
        dataframe.to_csv(f"./embeddings/{self.pdf_name}.csv", index=False)
        return dataframe

    def __insert_to_db(self, df):
        res = self.db_client.query("SHOW TABLES").named_results()
        tables = [r["name"] for r in res]
        # check if the table already exists
        if "embed_store" in tables:
            print("Table embed_store already exists...")
            return

        # if table does not exist already, create a table in the database
        print("Creating table embed_store in database...")
        self.db_client.command(
            """
            CREATE TABLE default.embed_store (
                id Int64,
                page_content String,
                embeddings Array(Float32),
                CONSTRAINT check_data_length CHECK length(embeddings) = 768
            ) ENGINE = MergeTree()
            ORDER BY id
        """
        )

        batch_size = 10
        num_batches = len(df) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_data = df[start_idx:end_idx]
            # insert the data to the table
            self.db_client.insert(
                "default.embed_store",
                batch_data.to_records(index=False).tolist(),
                column_names=batch_data.columns.tolist(),
            )
            print(f"Batch {i+1}/{num_batches} inserted.")
        # create a vector index for a quick retrieval of data
        self.db_client.command(
            """
        ALTER TABLE default.embed_store
            ADD VECTOR INDEX vector_index embeddings
            TYPE MSTG
        """
        )

    def __get_relevant_docs(self, user_query):
        print("Retrieving top 3 relevant chunks...")
        # call the get_embeddings function again to get the embeddings for the user query
        query_embeddings = self.__get_embeddings(user_query)
        results = self.db_client.query(
            f"""
            SELECT page_content,
            distance(embeddings, {query_embeddings}) as dist FROM default.embed_store ORDER BY dist LIMIT 3
        """
        )
        relevant_docs = []
        for row in results.named_results():
            relevant_docs.append(row["page_content"])
        return relevant_docs

    def __generate_prompt(self, query, relevant_passage):
        relevant_passage = " ".join(relevant_passage)
        prompt = (
            f"You are a helpful and informative chatbot that answers questions using text from the reference passage included below. "
            f"Respond in a complete sentence and make sure that your response is easy to understand for everyone. "
            f"Maintain a friendly and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
            f"QUESTION: '{query}'\n"
            f"PASSAGE: '{relevant_passage}'\n\n"
            f"ANSWER:"
        )
        return prompt

    def __generate_response(self, user_prompt):
        print("Generating response...")
        model = genai.GenerativeModel(MODEL_ID)
        answer = model.generate_content(user_prompt)
        return answer.text

    def save_response(self, query, response):
        print("Saving query and response to a file...")
        # save query and response to a file
        response_dir = "./response"
        if not os.path.exists(response_dir):
            os.makedirs(response_dir)

        # get the highest existing query number
        existing_files = [f for f in os.listdir(response_dir) if f.startswith("query_")]
        next_num = 1
        if existing_files:
            max_num = max([int(f.split("_")[1].split(".")[0]) for f in existing_files])
            next_num = max_num + 1

        # format and save the content
        with open(f"{response_dir}/query_{next_num}.txt", "w") as f:
            f.write(f"Query:\n{query}\n\nResponse:\n{response}")

    def query_chatbot(self, query):
        # prepare the data
        docs = self.__load_and_split()
        dataframe = self.__generate_embeddings(docs)
        self.__insert_to_db(dataframe)
        # get the relevant text
        relevant_text = self.__get_relevant_docs(query)
        text = " ".join(relevant_text)
        # generate the prompt and response
        prompt = self.__generate_prompt(query, relevant_text)
        answer = self.__generate_response(prompt)
        return answer


if __name__ == "__main__":
    pdf_path = "./pdfs/quaternions.pdf"
    chatbot = PDFChatbot(pdf_path)
    query_1 = "What is the research question of the paper?"
    query_2 = "A large percentage of these computerized machines are what?"
    response = chatbot.query_chatbot(query_2)
    chatbot.save_response(query_2, response)
    print(f"RAGBOT: {response}")
