import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Carrega variáveis de ambiente
load_dotenv()

# 1. Carregar o PDF
print("Carregando PDF...")
loader = PyPDFLoader("Documento.pdf")
docs = loader.load()

# 2. Divide o texto em partes (Chunks)
print("Dividindo o texto...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
splits = text_splitter.split_documents(docs)

print(f"Processamento concluído: {len(splits)} fragmentos criados.")

# 3. Salva no banco de Dados Local
print("Gerando banco de dados local... (isso pode levar alguns minutos)")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print(f"Sucesso! Banco de dados criado com {vectorstore._collection.count()} vetores.")