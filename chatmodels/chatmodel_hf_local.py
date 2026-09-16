from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
import torch

print("GPU:", torch.cuda.get_device_name(0))


os.environ['HF_HOME'] = "D:/models/hf"
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=0,
    pipeline_kwargs=dict(temperature=0.5,
                         max_new_tokens=100)
)
model = ChatHuggingFace(llm=llm)

hf_model = ChatHuggingFace(llm=llm)
result = hf_model.invoke("Tell me about India in 300 words.")
print(result.content)
