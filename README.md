# Cocktail-Advisor-Chat-Bot

A Python-based chat application that utilizes Retrieval-Augmented Generation (RAG) with a large language model to answer questions about cocktails and provide recommendations based on user preferences.

## Overview

This application combines a FastAPI backend with a simple HTML frontend to create an interactive chat system that:

1. Answers questions about cocktails using a cocktail dataset
2. Remembers user preferences for ingredients and cocktails
3. Provides recommendations based on those preferences
4. Uses a vector database to enable semantic search and similarity matching

The system uses RAG (Retrieval-Augmented Generation) to enhance LLM responses with relevant cocktail information stored in a vector database.

## Features

- **Knowledge Base Functionality**: Answer questions about cocktails, ingredients, and preparation methods
- **Advisor Functionality**: Recommend cocktails based on user preferences and similarity to other cocktails
- **Memory Management**: Detect and store user preferences in the vector database
- **Vector Search**: Find semantically similar cocktails and relevant information
- **Simple Web Interface**: Chat interface with example questions and responsive design

## Technical Components

- **FastAPI**: REST API backend with endpoints for chat and data retrieval
- **Vector Database**: FAISS implementation for storing and retrieving cocktail data and user preferences
- **Hugging Face Embeddings**: Document embeddings using the "all-MiniLM-L6-v2" model
- **LangChain Integration**: Framework for connecting the LLM with the vector database
- **RAG Implementation**: Custom retrieval and context enhancement for LLM queries

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ibishovrauf/Cocktail-Advisor-Chat-Bot.git
cd Cocktail-Advisor-Chat-Bot
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file in the root directory and add your environment variables:
```bash
# Create .env file
touch .env  # On Windows: type nul > .env

# Add the following to your .env file
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
python main.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Begin chatting with the cocktail assistant!

## Example Queries

### Knowledge Base Queries:
- What are the 5 cocktails containing lemon?
- What are the 5 non-alcoholic cocktails containing sugar?
- What are the ingredients in a Mojito?
- How do I make a Martini?

### Advisor Queries:
- My favorite ingredients are lime, rum, and mint
- Recommend 5 cocktails that contain my favorite ingredients
- Recommend a cocktail similar to "Hot Creamy Bush"
- What cocktail should I make if I like sweet drinks?

## Project Structure

```
cocktail_rag/
├── app/
│   ├── api.py            
│   ├── data_processing.py
│   ├── rag_engine.py
│   └── memory.py
├── data/
│   └── final_cocktails.csv
├── templates/
│   └── chat.html
├── requirements.txt
└── main.py
```

## Optimizations

The application uses several optimizations for better performance:

1. **Document Chunking**: Cocktail data is split into multiple chunks (general info, ingredients, instructions) for more targeted retrieval
2. **Query-Type Detection**: The system attempts to determine the most relevant document type based on query content
3. **Metadata Filtering**: Uses document type and cocktail ID metadata to filter and group relevant information
4. **Context Window Management**: Efficiently combines related document chunks to maximize context quality for the LLM

## Acknowledgments

- Cocktail dataset from [Kaggle](https://www.kaggle.com/datasets/aadyasingh55/cocktails)
- Built with [LangChain](https://github.com/hwchase17/langchain) and [FAISS](https://github.com/facebookresearch/faiss)
