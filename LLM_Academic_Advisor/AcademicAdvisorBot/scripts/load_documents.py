import os
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_docs(path: str):
    if os.path.isdir(path):
        print(f"📂 Loading directory: {path}")
        loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    elif os.path.isfile(path) and path.endswith(".pdf"):
        print(f"📄 Loading single PDF file: {path}")
        loader = PyMuPDFLoader(path)
    else:
        raise ValueError(f"Invalid path: {path}. Must be a directory or PDF file.")

    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)
