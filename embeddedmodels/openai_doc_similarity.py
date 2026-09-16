from langchain_openai import OpenAIEmbeddings
import torch
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch.xpu import device

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)
documents = [
    "Python is a popular programming language used for web development, automation, data analysis, and artificial intelligence.",

    "Machine learning enables computers to learn patterns from data and make predictions without being explicitly programmed for every situation.",

    "Vector embeddings represent text as numerical vectors. Similar meanings are represented by vectors that are close to each other in vector space.",

    "Semantic search finds documents based on meaning rather than simply matching the exact words used in the query.",

    "Docker packages applications and their dependencies into containers, making software easier to deploy consistently across different environments.",

    "Kubernetes is a container orchestration platform that manages deployment, scaling, networking, and availability of containerized applications.",

    "PostgreSQL is a relational database system commonly used for storing structured application data and running complex SQL queries.",

    "Redis is an in-memory data store frequently used for caching, session management, queues, and fast key-value lookups.",

    "A REST API allows different software systems to communicate over HTTP using operations such as GET, POST, PUT, and DELETE.",

    "Authentication verifies the identity of a user, while authorization determines what resources or actions that user is allowed to access.",

    "CI/CD pipelines automate software building, testing, and deployment so that changes can be delivered to production more reliably.",

    "Monitoring systems collect metrics, logs, and traces to help engineers detect failures and understand application performance.",

    "A message queue allows applications to communicate asynchronously by placing messages into a queue that can be processed by consumers.",

    "Data engineering focuses on collecting, transforming, storing, and delivering data so that it can be reliably used for analytics and machine learning.",

    "Idempotency means that performing the same operation multiple times produces the same final result as performing it once.",

    "Caching improves application performance by storing frequently accessed data in a faster storage layer so it does not need to be recomputed or retrieved repeatedly.",

    "Load balancing distributes incoming network traffic across multiple servers to improve availability and prevent individual servers from becoming overloaded.",

    "A database index speeds up data retrieval by creating an optimized data structure that allows the database to locate records without scanning the entire table.",

    "An embedding model converts text, images, or other information into numerical vectors that capture semantic characteristics.",

    "Retrieval-augmented generation combines information retrieval with a language model so that generated answers can be grounded in relevant external documents."
]

query = "How can I search documents based on their meaning instead of matching exact words?"


doc_vectors = embedding.embed_documents(documents)
query_vectors = embedding.embed_query(query)

similarities = cosine_similarity([query_vectors], doc_vectors)
scores = similarities[0]

index, score, = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]
print(query)
print(documents[index])
print("similarities score is:", score)
