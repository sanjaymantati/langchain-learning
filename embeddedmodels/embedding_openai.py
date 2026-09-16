from langchain_openai import OpenAIEmbeddings

from dotenv import  load_dotenv


load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

vector = embedding.embed_query("This is the first sentence to embed.")
print(str(vector))
