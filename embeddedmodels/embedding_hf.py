from langchain_huggingface import  HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
text = "This is the first sentence to embed."

embedding = embedding.embed_query(text)
print(str(embedding))