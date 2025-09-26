# Install dependencies first:
# pip install streamlit transformers torch

import streamlit as st
from transformers import pipeline
import textwrap

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="SmartStudy", page_icon="📘", layout="centered")
st.title("📘 SmartStudy - College Topic Helper")
st.markdown("Ask about **college topics like binary, database, photosynthesis, etc.**")

# ---------------- Study Topics Data ----------------
study_data = {
    "Binary": "Binary is a number system that uses only two digits, 0 and 1. It is fundamental in computing since computers use binary to represent data and instructions. Each binary digit is called a bit. Binary allows computers to perform calculations and store information efficiently.",
    "Database": "A database is a structured collection of data that allows easy access, management, and updating. Databases are used in almost every application to store user data, transactions, or other relevant information. Modern databases use query languages like SQL to manage data. They ensure data integrity, security, and efficient retrieval.",
    "Photosynthesis": "Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into glucose and oxygen. It is the primary source of energy for plants and indirectly for other living organisms. Chlorophyll in the leaves absorbs sunlight to drive the chemical reactions. This process is vital for life on Earth."
}

# ---------------- Load Transformer Model ----------------
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2", device=-1)

generator = load_model()

# ---------------- Sidebar ----------------
st.sidebar.title("📌 Quick Topics")
for topic in study_data.keys():
    st.sidebar.markdown(f"- {topic}")

# ---------------- Main Input ----------------
query = st.text_input("💬 Type your topic here:")

if st.button("Get Explanation") and query:
    # Check if topic exists in data
    if query in study_data:
        answer = study_data[query]
    else:
        # Generate answer using transformer
        prompt = f"Explain the topic '{query}' in 3 sections: Definition, Key Points, Real World Example."
        response = generator(
            prompt,
            max_length=300,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=50256
        )
        answer = response[0]["generated_text"]

    # Wrap text
    wrapped = "\n".join([textwrap.fill(line, width=90) for line in answer.split("\n")])

    # Display
    st.subheader("🔍 Explanation")
    st.text(wrapped)

    # Save to file
    with open("study_notes.txt", "a", encoding="utf-8") as f:
        f.write(f"\n\n### Topic: {query}\n{wrapped}\n{'-'*80}\n")

    st.success("📝 Saved to study_notes.txt")
