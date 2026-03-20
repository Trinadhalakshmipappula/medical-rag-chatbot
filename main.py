from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import CrossEncoder
import faiss
import os

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
# 4. Load Models
# -----------------------------
print("Loading models...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True}
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# -----------------------------
# 5. Load PDFs
# -----------------------------
FILE_PATH = "medical_data"

loader = DirectoryLoader(
    FILE_PATH,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()
print("Loaded docs:", len(docs))

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embedding_dim = 768
index = faiss.IndexFlatIP(embedding_dim)

vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store.add_documents(split_docs)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

print("RAG ready!")

# -----------------------------
# 6. Re-ranking
# -----------------------------
def rerank(query, docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]

# -----------------------------
# 7. Groq Answer
# -----------------------------
def generate_answer(query, context):
    try:
        prompt = f"""
Context:
{context}

Question:
{query}

Answer in 2 lines:
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

# -----------------------------
# 8. API Endpoint
# -----------------------------
@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        query = request.query

        docs = retriever.invoke(query)

        if not docs:
            return {"answer": "No data found"}

        final_docs = rerank(query, docs)

        context = "\n\n".join([doc.page_content for doc in final_docs])

        answer = generate_answer(query, context)

        return {
            "query": query,
            "answer": answer
        }

    except Exception as e:
        return {"error": str(e)}