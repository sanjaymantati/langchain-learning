from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)
docs = [
    "This is the first sentence to embed.",
    "This is the second sentence to embed."
    "This is the third sentence to embed."
]
vector = embedding.embed_documents(docs)
print(str(vector))
