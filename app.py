import streamlit as st
import requests

st.set_page_config(page_title="Medical RAG Bot")

# -----------------------------
# 🎨 Custom Styling
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #e0f7fa, #e8f5e9);
}

/* Remove ALL default outlines globally */
* {
    outline: none !important;
}

/* Input box */
.stTextInput input {
    border: 2px solid #4CAF50 !important;
    border-radius: 10px;
    padding: 10px;
    background-color: white;
    box-shadow: none !important;
}

/* Input focus (force green only) */
.stTextInput input:focus,
.stTextInput input:focus-visible {
    border: 2px solid #4CAF50 !important;
    box-shadow: 0 0 6px #4CAF50 !important;
    outline: none !important;
}

/* Remove red error styles */
input:invalid {
    box-shadow: none !important;
    border: 2px solid #4CAF50 !important;
}

/* Button */
.stButton button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none !important;
    outline: none !important;
}

/* Button hover */
.stButton button:hover {
    background-color: #388e3c;
}

/* Button focus/click fix */
.stButton button:focus,
.stButton button:focus-visible,
.stButton button:active {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
    background-color: #2e7d32 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# Title
# -----------------------------
st.markdown(
    "<h1 style='text-align: center;'>🩺 MedQuery AI – RAG-based Medical Chatbot</h1>",
    unsafe_allow_html=True
)

# -----------------------------
# Input
# -----------------------------
query = st.text_input("Ask your medical question:")

# -----------------------------
# Button Action
# -----------------------------
if st.button("Ask"):
    if query:
        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={"query": query}
        )

        if response.status_code == 200:
            data = response.json()
            st.subheader("Answer:")
            st.write(data["answer"])
        else:
            st.error("Error connecting to backend")