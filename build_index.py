# build_index.py

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

loader = DirectoryLoader("medical_data", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db = FAISS.from_documents(split_docs, embeddings)
db.save_local("faiss_index")

print("✅ Index created")
docs = loader.load()
print("Total docs loaded:", len(docs))
print("Total chunks:", len(split_docs))