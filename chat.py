from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Carregar o modelo de Busca (Embeddings)
print("Carregando busca local...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Carregar o Banco de Dados
print("Carregando banco de dados...")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Definir o Cérebro (LLM) usando Ollama
print("Carregando o cérebro Llama 3 (Ollama)...")
llm = ChatOllama(model="llama3", temperature=0)

# 4. Criar o Prompt
system_prompt = (
    "Você é um assistente útil. Use os seguintes trechos de contexto recuperados "
    "para responder à pergunta. Se você não souber a resposta com base no contexto, "
    "diga que não sabe. Mantenha a resposta concisa."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Função para formatar os documentos recuperados em texto
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. Montar a Cadeia usando o operador Pipe (|) - Método LCEL
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Loop de Conversa
print("\n--- Chatbot Local Iniciado! (Digite 'sair' para encerrar) ---")
while True:
    pergunta = input("\nVocê: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        break
    
    # Rodar a chain
    print("Robô: processando...")
    resposta = rag_chain.invoke(pergunta)
    
    print(f"Robô: {resposta}")