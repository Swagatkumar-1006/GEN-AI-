import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from huggingface_hub import InferenceClient
import tempfile
import numpy as np

st.set_page_config(page_title="PDF Q&A Bot with Qwen", layout="centered")
st.title("PDF Q&A Bot with Qwen 1.5B (Hugging Face API)")
st.write("Upload a PDF and ask questions. Powered by Qwen2-1.5B-Instruct.")

hf_token = os.getenv("huggingfacetoken")  
if not hf_token:
    st.error("Please set your Hugging Face API token in the 'huggingfacetoken' environment variable.")
    st.stop()

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.texts = None

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file and st.session_state.vector_store is None:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(texts, embeddings)

        st.session_state.vector_store = vector_store
        st.session_state.texts = texts

        os.unlink(tmp_file_path)
        st.success("PDF processed successfully!")
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")

if st.session_state.vector_store is not None:
    query = st.text_input("Ask a question about the PDF:")
    if query:
        try:
            embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            query_embedding = embedder.embed_query(query)

            D, I = st.session_state.vector_store.index.search(np.array([query_embedding]), k=3)
            context_chunks = [st.session_state.texts[i].page_content for i in I[0]]

            context = "\n\n".join(context_chunks)

            user_message = (
                "You are a helpful assistant. Use the following context to answer the question. "
                "If the answer is not in the context, say so clearly.\n\n"
                f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
            )

            client = InferenceClient(api_key=hf_token, provider="featherless-ai")

            completion = client.chat.completions.create(
                model="Qwen/Qwen2.5-1.5B-Instruct",
                messages=[{"role": "user", "content": user_message}],
            )

            answer = completion.choices[0].message["content"]

            st.markdown("**Answer:**")
            st.write(answer)

        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
else:
    if uploaded_file is None:
        st.info("Upload a PDF file to begin.")
    elif st.session_state.vector_store is None:
        st.warning("Processing the PDF. Please wait...")
