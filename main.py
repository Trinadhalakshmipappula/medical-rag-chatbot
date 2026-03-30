from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# 1. Configure Groq API
# -----------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# 2. FastAPI App
# -----------------------------
app = FastAPI(title="Medical RAG Bot with Groq")

# -----------------------------
# 3. Request Schema
# -----------------------------
class QueryRequest(BaseModel):
    query: str

# -----------------------------
# 4. Load ONLY embeddings + FAISS (no building)
# -----------------------------
print("Loading FAISS index...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vector_store = FAISS.load_local(
    "faiss_index",   # folder you must create locally
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

print("RAG ready!")

# -----------------------------
# 5. Groq Answer
# -----------------------------
def generate_answer(query, context):
    try:
        prompt = f"""
You are a medical assistant.

STRICT RULES:
- Answer ONLY from the given context
- If the answer is NOT in the context, say: "No data found"
- Do NOT use outside knowledge

Context:
{context}

Question:
{query}

Give a detailed answer in 4-6 lines:
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# 6. API Endpoint
# -----------------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        query = request.query

        docs = retriever.invoke(query)

        # ✅ Check if no docs found
        if not docs or len(docs) == 0:
            return {"answer": "No data found"}

        context = "\n\n".join([doc.page_content for doc in docs])

        answer = generate_answer(query, context)

        return {
            "query": query,
            "answer": answer
        }

    except Exception as e:
        return {"error": str(e)}