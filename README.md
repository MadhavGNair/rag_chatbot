# RAG Chatbot

This repository contains a RAG (Retrieval-Augmented Generation) based chatbot implementation.

## Project Structure

```
rag_chatbot/
    ├──chatbot-cdk                      # AWS CDK implementation
        ├──chatbot_cdk
            ├──chatbot_cdk_stack.py     # contains the CDK stack that defines the Lambda function
    ├──src
        ├──app                          # contains the FastAPI implementation
        ├──api.py                    # defines function to initialize and query chatbots
        ├──main.py                   # defines the app, handler, and endpoints
        ├──models.py                 # initializes BaseModels with default parameters
        ├──rag_chatbot                  # contains the core chatbot implementation
            ├──pdfs                     # raw PDFs used as knowledge base for chatbot
            ├──pdf_chatbot.py           # defines the core chatbot class
        ├──Dockerfile                   
    ├──example.env                      # add all required API keys and rename to ".env" 
    ├── README.md                       # documentation file (you are here)
```

## About

This is a RAG-based chatbot project that leverages retrieval-augmented generation techniques to provide more accurate and context-aware responses. The models used are as follows,

- Embedding model - text-embedding-3-small (OpenAI)
- LLMs - OpenAI, Gemini, Claude, AWS Bedrock (supports all models available through API- specify them in /init)

## Future Developments

1. Enhance vanilla-RAG techniques with BM25, OP-RAG, and so on.
2. Develop a web interface
3. Allow async calls and batched calls

## Author

Created by Madhav Girish Nair (madhavgirish02@gmail.com)

 
