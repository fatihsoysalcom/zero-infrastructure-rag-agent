import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# --- Configuration ---
# This example uses a local Ollama instance for the LLM and HuggingFace for embeddings.
# Ensure you have Ollama installed and a model like 'llama3' pulled:
# ollama pull llama3
# Ensure you have the 'langchain-community' and 'langchain-huggingface' libraries installed:
# pip install langchain-community langchain-huggingface

# --- Mock Knowledge Base ---
# In a real scenario, this would be loaded from files, databases, etc.
KNOWLEDGE_BASE_FILE = "knowledge_base.txt"

def create_mock_knowledge_base():
    content = """
    Yapay zeka (AI), bilgisayarların insan benzeri zeka sergilemesini sağlayan bir bilgisayar bilimi alanıdır.
    Büyük dil modelleri (LLM'ler), metin verileri üzerinde eğitilmiş ve doğal dil anlama ve üretme yeteneklerine sahip AI modelleridir.
    Geri Çağırma Artırılmış Üretim (RAG), LLM'lerin harici bilgi kaynaklarından bilgi alarak yanıtlarını zenginleştirmesini sağlayan bir tekniktir.
    Çok Modlu Bağlam İşleme (MCP), metin, görüntü, ses gibi farklı veri türlerini işleyebilen sistemleri ifade eder.
    Sıfır altyapılı RAG ajanları, karmaşık altyapı yönetimi gerektirmeden RAG yetenekleri sunar.
    DigitalOcean, bulut bilişim hizmetleri sunan bir şirkettir.
    """
    with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
        f.write(content)

create_mock_knowledge_base()

# --- RAG Pipeline Setup ---

# 1. Load and Split Documents
documents = TextLoader(KNOWLEDGE_BASE_FILE, encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 2. Create Embeddings
# Using a local HuggingFace model for embeddings. Requires 'sentence-transformers' installed.
# pip install sentence-transformers
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 3. Create Vector Store (FAISS for local, in-memory)
vectorstore = FAISS.from_documents(chunks, hf_embeddings)
retriever = vectorstore.as_retriever()

# 4. Define LLM (using Ollama for local, zero-install LLM)
# Ensure Ollama is running and 'llama3' model is pulled.
llm = Ollama(model="llama3")

# 5. Define Prompt Template
# This prompt guides the LLM to use the retrieved context.
prompt_template = """
Soru: {question}

Bağlam:
{context}

Yukarıdaki bağlamı kullanarak soruyu yanıtlayın. Eğer cevabı bağlamda bulamazsanız, "Bilmiyorum" deyin.
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# 6. Create RAG Chain
# This chain combines retrieval, prompt formatting, and LLM generation.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# --- Example Usage ---
def ask_rag_agent(question):
    print(f"\nSoru: {question}")
    response = rag_chain.invoke(question)
    print(f"Yanıt: {response}")

if __name__ == "__main__":
    print("Sıfır Altyapılı RAG Ajanı Başlatıldı...")
    print("Bilgi tabanı oluşturuldu ve indekslendi.")

    # Test questions
    ask_rag_agent("Yapay zeka nedir?")
    ask_rag_agent("RAG teknolojisi ne işe yarar?")
    ask_rag_agent("MCP ne anlama gelir?")
    ask_rag_agent("DigitalOcean ne sunar?")
    ask_rag_agent("Python programlama dili hakkında bilgi ver.") # This should result in 'Bilmiyorum'

    # Clean up mock file
    os.remove(KNOWLEDGE_BASE_FILE)
