# GEN-AI-PROJECT
PDF Q&A Bot with Qwen
A Streamlit-based web application that allows users to upload a PDF file (e.g., lecture notes, manuals) and ask questions about its content. The app processes the PDF, creates a vector store of its text, and uses a language model to provide relevant answers based on the document's content. If the answer is not found in the PDF, the app provides a general answer using the Qwen2-1.5B-Instruct model. Powered by LangChain, FAISS, Hugging Face's Qwen2-1.5B-Instruct model, and Streamlit.


Features

Upload a PDF file and process its content for question answering.
Ask natural language questions about the PDF content.
Uses FAISS for efficient vector storage and retrieval of text chunks.
Leverages Hugging Face's Qwen2-1.5B-Instruct model to generate answers based on PDF content or provide general answers if the information is not in the PDF.
Simple and intuitive user interface built with Streamlit.

Prerequisites

Python 3.8 or higher
A Hugging Face API token (set as the huggingfacetoken environment variable)
A compatible PDF file for processing

Installation

Clone the repository:
git clone https://github.com/Swagatkumar-1006/GEN-AI-.git
cd pdf-qa-bot


Create and activate a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install the required dependencies:
pip install -r requirements.txt


Set up your Hugging Face API token as an environment variable:
export huggingfacetoken='your-hugging-face-api-token'  # On Windows: set huggingfacetoken=your-hugging-face-api-token



Dependencies
The app relies on the following Python packages (listed in requirements.txt):
streamlit
langchain
langchain-community
faiss-cpu
huggingface_hub
sentence-transformers
PyPDF2
numpy

Install them using:
pip install streamlit langchain langchain-community faiss-cpu huggingface_hub sentence-transformers PyPDF2 numpy

Usage

Run the Streamlit app:
streamlit run app.py


Open your browser 

Upload a PDF file using the file uploader.

Once the PDF is processed, enter a question about the PDF content in the text input field.

View the AI-generated answer, which will be based on the PDF content if the information is present, or a general answer if the information is not found in the PDF.


How It Works

PDF Processing:

The uploaded PDF is saved temporarily and loaded using PyPDFLoader from LangChain.
The text is split into chunks using RecursiveCharacterTextSplitter (chunk size: 1000, overlap: 200).


Vector Store:

Text chunks are embedded using the sentence-transformers/all-MiniLM-L6-v2 model from Hugging Face.
Embeddings are stored in a FAISS vector store for efficient similarity search.


Question Answering:

The user's query is embedded and used to retrieve the top 3 relevant text chunks from the vector store.
If relevant chunks are found, they are combined into a context and sent to the Qwen2-1.5B-Instruct model via the Hugging Face Inference API to generate an answer based on the PDF content.
If the answer is not found in the PDF, the Qwen2-1.5B-Instruct model provides a general answer based on its knowledge.
The model generates an answer based on the provided context and question.


Error Handling:

The app checks for a valid Hugging Face API token and handles errors during PDF processing or answer generation.
Temporary files are cleaned up after processing.



Notes

Ensure your Hugging Face API token is set correctly in the environment variable huggingfacetoken.
The app uses the sentence-transformers/all-MiniLM-L6-v2 model for embeddings, which is lightweight and efficient.
The Qwen2-1.5B-Instruct model is hosted on Featherless AI (via Hugging Face's Inference API) for fast inference.
The app stores the vector store and text chunks in Streamlit's session_state to avoid reprocessing the PDF for each question.
