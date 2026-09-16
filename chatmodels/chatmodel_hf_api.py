from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation"
)

hf_model = ChatHuggingFace(llm=llm)
result = hf_model.invoke("What is the capital of France?")
print(result.content)
