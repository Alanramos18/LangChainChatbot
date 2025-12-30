import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

def load_pdfs_from_folder(folder_path: str):
    documents = []
    folder = Path(folder_path)
    for pdf_file in folder.rglob("*.pdf"):
        print(f"📄 Leyendo: {pdf_file.name}")
        subject = pdf_file.parent.name
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        for doc in docs:
            doc.metadata["subject"] = subject
            doc.metadata["file_name"] = pdf_file.name
        documents.extend(docs)
    return documents

def split_documents(documents):
    documents = [d for d in documents if is_valid_text(d.page_content)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if is_valid_text(c.page_content)]
    
    return chunks

def create_faiss_index(chunks):
    load_dotenv()
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local("faiss_index")
    print("✅ FAISS index guardado en disco")
    return index


def is_valid_text(text: str) -> bool:
    return bool(re.search(r"\w", text))

if __name__ == "__main__":
    docs = load_pdfs_from_folder("docs")
    chunks = split_documents(docs)
    print(f"🔹 Chunks generados: {len(chunks)}")
    faiss_index = create_faiss_index(chunks)