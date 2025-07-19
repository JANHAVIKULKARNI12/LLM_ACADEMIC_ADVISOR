from load_documents import load_and_split_docs

docs = load_and_split_docs()

print(f"✅ Loaded {len(docs)} chunk(s)")
print("📄 Preview of first chunk:")
print(docs[0].page_content[:500])
