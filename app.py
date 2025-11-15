# app.py
import streamlit as st
from transformers import pipeline

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Summarizer & QA", layout="wide")
st.title("⚡ Article Summarizer & Generative QA")

# ===== Load models =====
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_qa_model():
    # Use a generative model suitable for longer text
    return pipeline("text2text-generation", model="google/flan-t5-base")

summarizer = load_summarizer()
qa_model = load_qa_model()

# ===== Text Input =====
ARTICLE = st.text_area(
    "Enter text:", 
    height=400, 
    placeholder="Paste or type your article here..."
)

# ===== Summarization =====
if st.button("Summarize"):
    if ARTICLE.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        summary = summarizer(
            ARTICLE, 
            min_length=150, 
            max_length=3000,  # detailed summary
            do_sample=False
        )
        st.subheader("Summary:")
        st.write(summary[0]['summary_text'])

# ===== Generative Question Answering =====
st.subheader("Ask a question about the article")
QUESTION = st.text_input("Enter your question:")

if st.button("Answer Question"):
    if ARTICLE.strip() == "":
        st.warning("Please enter an article first.")
    elif QUESTION.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Directly use the full article as context
        input_text = f"Answer the question in detail based on the context:\nContext: {ARTICLE}\nQuestion: {QUESTION}"
        answer = qa_model(input_text, max_length=500)[0]['generated_text']
        st.subheader("Answer:")
        st.write(answer)
