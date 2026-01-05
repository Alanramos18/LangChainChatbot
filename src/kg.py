from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import networkx as nx

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
faiss_index = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def extract_entities_and_relationships(chunk_text):
    prompt = f"""
    Extrae entidades y relaciones del siguiente texto.
    Formato: {"entidad1","relación","entidad2"} por línea
    Texto:
    {chunk_text}
    """
    resultado = llm.invoke(prompt).content
    relaciones = []
    for line in resultado.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            relaciones.append(parts)
    return relaciones

G = nx.DiGraph()

for doc_id in faiss_index.index_to_docstore_id.values():
    doc = faiss_index.docstore.search(doc_id)
    relaciones = extract_entities_and_relationships(doc.page_content)
    for ent1, rel, ent2 in relaciones:
        G.add_node(ent1)
        G.add_node(ent2)
        G.add_edge(ent1, ent2, relation=rel)