# Install dependencies first:
# pip install streamlit langchain-huggingface langchain-community faiss-cpu sentence-transformers transformers requests

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import streamlit as st

# ---------------- Study Topics Data ----------------
study_data = [
    {"topic": "Binary", "explanation": "Binary is a number system that uses only two digits, 0 and 1. It is fundamental in computing since computers use binary to represent data and instructions. Each binary digit is called a bit. Binary allows computers to perform calculations and store information efficiently."},
    {"topic": "Database", "explanation": "A database is a structured collection of data that allows easy access, management, and updating. Databases are used in almost every application to store user data, transactions, or other relevant information. Modern databases use query languages like SQL to manage data. They ensure data integrity, security, and efficient retrieval."},
    {"topic": "Photosynthesis", "explanation": "Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into glucose and oxygen. It is the primary source of energy for plants and indirectly for other living organisms. Chlorophyll in the leaves absorbs sunlight to drive the chemical reactions. This process is vital for life on Earth."}
]

# ---------------- Prepare Documents ----------------
docs = [Document(page_content=f["explanation"], metadata={"topic": f["topic"]}) for f in study_data]

# ---------------- Embeddings & Vector Store ----------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# ---------------- LLM Setup ----------------
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# ---------------- RetrievalQA Chain ----------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    return_source_documents=True
)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="SmartStudy", page_icon="📘", layout="centered")

st.title("📘 SmartStudy - College Topic Helper")
st.markdown("Ask anything about **college topics like binary, database, photosynthesis, etc.**")

# Sidebar Quick Topics
st.sidebar.title("📌 Quick Topics")
quick_topics = [t["topic"] for t in study_data]
for item in quick_topics:
    st.sidebar.markdown(f"- {item}")

# Main Input
query = st.text_input("💬 Type your topic or question here:")

if st.button("Get Explanation") and query:
    result = qa_chain(query)
    answer = result["result"]
    source_docs = result.get("source_documents", [])

    st.subheader("🔍 Explanation")
    st.text(answer)

    if source_docs:
        matched_topic = source_docs[0].metadata["topic"]
        st.info(f"📌 Based on topic: *{matched_topic}*")

    # Save explanation to file
    with open("study_notes.txt", "a", encoding="utf-8") as f:
        f.write(f"\n\n### Topic: {query}\n{answer}\n{'-'*80}\n")

    st.success("📝 Saved to study_notes.txt")
