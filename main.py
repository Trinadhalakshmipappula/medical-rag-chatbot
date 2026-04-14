from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("GROQ KEY LOADED:", GROQ_API_KEY is not None)

# -----------------------------
# 2. FastAPI App
# -----------------------------
app = FastAPI(title="Medical RAG Bot with Groq")

# -----------------------------
# 3. Groq Client (FIXED)
# -----------------------------
client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# 4. Request Schema
# -----------------------------
class QueryRequest(BaseModel):
    query: str

# -----------------------------
# 5. Load Embeddings + FAISS
# -----------------------------
print("Loading FAISS index...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vector_store = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

print("RAG ready!")

# -----------------------------
# 6. Generate Answer (Groq)
# -----------------------------
def generate_answer(query, context):
    try:
        prompt = f"""
You are a medical assistant.

STRICT RULES:
- Answer ONLY using the given context
- If answer is not in context, say "No data found"
- Keep answer clear and simple (4-6 lines)

Context:
{context}

Question:
{query}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# 7. API Routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        query = request.query

        # Retrieve relevant documents
        docs = vector_store.similarity_search(query, k=5)
        
        print("Retrieved docs:", len(docs))

        if len(docs) == 0:
            return {"answer": "No data found"}

        context = "\n\n".join([doc.page_content for doc in docs])

        answer = generate_answer(query, context)

        return {
            "query": query,
            "answer": answer
        }

    except Exception as e:
        return {"error": str(e)}