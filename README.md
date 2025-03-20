# Multi-PDF Chatbot

A Streamlit-based application that allows users to chat with multiple PDF documents using AI. This tool processes uploaded PDFs, creates vector embeddings, and answers questions based on the content of the documents.

## Features

- Upload multiple PDF documents
- Process and extract text from PDFs
- Create vector embeddings for efficient retrieval
- Ask questions about the content of the PDFs
- Get AI-generated responses based on document content

## Screenshots

### Main Interface
![image](https://github.com/user-attachments/assets/2b54ef03-8efa-41e3-8adc-fa539386b0c7)

### PDF Upload Process
![image](https://github.com/user-attachments/assets/50a6099b-0052-4dac-abf9-21d69dfeb1f2)

### Question Answering
![image](https://github.com/user-attachments/assets/188c041f-ca1f-4fe7-8e7c-01af216b3802)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/thegupta1694/ChatPDF.git
   cd ChatPDF
   ```

2. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided URL (usually http://localhost:8501)

3. Upload your PDF files using the sidebar

4. Click "Submit & Process" to extract text and create embeddings

5. Ask questions about the content of the PDFs in the text input field

## How It Works

1. **PDF Processing**: The app extracts text from uploaded PDFs using PyPDF2
2. **Text Chunking**: The extracted text is split into manageable chunks using LangChain's RecursiveCharacterTextSplitter
3. **Vector Embeddings**: The text chunks are converted into vector embeddings using OpenAI's embedding model
4. **Vector Store**: The embeddings are stored in a FAISS vector database for efficient similarity search
5. **Question Answering**: When a user asks a question, the app retrieves the most relevant document chunks and uses OpenAI's GPT-4o-mini to generate an answer

## Dependencies

- Streamlit: For the web interface
- PyPDF2: For PDF text extraction
- LangChain: For text processing and LLM chains
- FAISS: For vector storage and similarity search
- OpenAI: For embeddings and LLM capabilities

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key

## Safety Note

The application uses FAISS with `allow_dangerous_deserialization=True` when loading the vector store. This is necessary for the current implementation but should be used with caution in production environments.

## Author

Arya Ajay Gupta

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [OpenAI](https://openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
