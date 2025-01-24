# RAG Chatbot

This repository contains a RAG (Retrieval-Augmented Generation) based chatbot implementation.

## Project Structure

```
rag_chatbot/
├──embeddings
    ├──.csv files    # Where embeddings of PDFs are cached to prevent regeneration
├──pdfs
    ├──.pdf files    # Where PDF files must be placed
├──pdf_chatbot.py    # The class definition for RAG chatbot
├──main.py           # The main file to execute desired querying
├── README.md        # Main documentation file (you are here)
```

## About

This is a RAG-based chatbot project that leverages retrieval-augmented generation techniques to provide more accurate and context-aware responses. The models used are as follows,

- Embedding model - text-embedding-004 (Gemini)
- LLM - gemini-2.0-flash-exp (Gemini)

## Getting Started

1. Clone the repository
2. Follow the setup instructions below
3. Start using the chatbot

## Setup

1. Create a .env file in the root directory (./rag_chatbot) and create a "GEMINI_API_KEY" instance with your API key ([Gemini API](https://aistudio.google.com/app/apikey)).
2. cd to root directory in your terminal
3. Run 'poetry shell' to spawn virtual environment
4. Run 'poetry install' to install dependencies
NOTE: change dependency versions at your own risk, most dependencies are version sensitive.

## Usage

1. Place the desired PDF under './rag_chatbot/pdfs'
If CLI interface is preferred, 
2. Run 'main.py' and follow instructions to query
else,
2. Edit the "query" parameter in 'main.py' and run the file
3. Responses are displayed on screen as well as saved to the './rag_chatbot/response' directory
    - the responses are saved in the format 'query_X.txt' where X indicates the order of querying with higher X indicating more recent queries and responses

## Future Developments

1. Implement advanced RAG techniques such as BM25, OP-RAG, and so on.
2. Develop a web interface
3. Test better embedding models
4. Allow async calls and batched calls

## Author

Created by Madhav Girish Nair (madhavgirish02@gmail.com)

 
