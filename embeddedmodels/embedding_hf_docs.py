from langchain_huggingface import  HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
text = "This is the first sentence to embed."
docs = [
    "This is the first sentence to embed.",
    "This is the second sentence to embed."
    "This is the third sentence to embed."
]
embedding = embedding.embed_documents(docs)
print(str(embedding))