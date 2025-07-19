import os
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from scripts.load_documents import load_and_split_docs
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

llm = ChatOpenRouter(
    model="mistralai/mistral-7b-instruct",  # change this if needed
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
)
load_dotenv()

def build_chain():
    print("🔧 Starting chain setup...")

    # 📄 Load PDF data
    docs = load_and_split_docs("AcademicAdvisorBot/data/iteration1_java.pdf")


    # 🧠 Embed using HuggingFace
    print("🧠 Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✅ Embeddings ready!")

    # 📦 Vector DB
    print("📦 Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    print("✅ Vectorstore ready.")

    # 🤖 Use OpenRouter-hosted Gemma via OpenAI-compatible interface
    print("🤖 Connecting to LLM...")
    llm = ChatOpenAI(
        model="google/gemma-3n-e2b-it",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=os.getenv("OPENROUTER_BASE_URL")
    )
    print("✅ LLM ready.")

    # 🔗 Retrieval QA
    print("🔗 Creating RetrievalQA Chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    print("✅ Chain ready!")

    return qa_chain
