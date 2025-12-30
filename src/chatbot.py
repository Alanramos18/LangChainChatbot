from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
faiss_index = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = faiss_index.as_retriever(search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

input = "¿Quienes son los profesores de Gestion Aplicada al Desarrollo de Software II?"
system_prompt = (
    "Eres un asistente útil y experto en la carrera de ingeniería en informática. "
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Usa el siguiente contexto para responder la pregunta: {context}\nPregunta: {input}"),
    ]
)

combine_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
qa = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combine_chain
)

respuesta = qa.invoke({"input": input})
respuesta_final = respuesta["answer"]

print("Pregunta:", input)
print("Respuesta:", respuesta_final)