import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG PDF Chat", layout="wide")

st.title("📚 Chat with your Book (RAG)")


if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        # Embeddings
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create DB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )

        st.session_state.vectorstore = vectorstore

        st.success(f"PDF processed! {len(chunks)} chunks created.")


if st.session_state.vectorstore:

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ])

    query = st.text_input("Ask something about the book:")

    if query:
        with st.spinner("Thinking..."):

            docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            response = llm.invoke(final_prompt)

            st.subheader("Answer:")
            st.write(response.content)

            with st.expander("🔍 Retrieved Context"):
                for i, doc in enumerate(docs):
                    st.write(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.write("---")

else:
    st.info("Upload a PDF to start chatting.")