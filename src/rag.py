import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import GoogleGeminiEmbeddings

load_dotenv()

DOCS_DIR = "../docs"

def load_documents():
    documents = []
    for file_name in os.listdir(DOCS_DIR):
        file_path = os.path.join(DOCS_DIR, file_name)
        if file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file_name.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)

def create_embeddings(docs):
    embeddings = GoogleGeminiEmbeddings(model="textembedding-gecko-001")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def query(vector_store, question, k=3):
    docs = vector_store.similarity_search(question, k=k)
    return "\n".join([d.page_content for d in docs])

def save_vector_store(vector_store, path="faiss_index"):
    vector_store.save_local(path)

def load_vector_store(path="faiss_index"):
    embeddings = GoogleGeminiEmbeddings(model="textembedding-gecko-001")
    return FAISS.load_local(path, embeddings)

# ===========================
# EJEMPLO DE USO
# ===========================
if __name__ == "__main__":
    print("📄 Cargando documentos...")
    docs = load_documents()
    # chunks = split_documents(docs)
    # print(f"📝 Documentos divididos en {len(chunks)} chunks.")

    # print("⚡ Creando embeddings con Gemini...")
    # vector_store = create_embeddings(chunks)
    # save_vector_store(vector_store)

    # print("🔎 Probando query de ejemplo...")
    # question = "¿Qué técnicas se usan para analizar datos?"
    # result = query(vector_store, question)
    # print("\n📌 Documentos relevantes encontrados:\n", result)