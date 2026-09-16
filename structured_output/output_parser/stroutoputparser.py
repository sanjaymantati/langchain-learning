from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

#ENDPOINT VERSION
# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

# LOCAL VERSION
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

# model = ChatHuggingFace(llm=llm)

## detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

## summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following report: {report}',
    input_variables=['report']
)

prompt1 = template1.invoke({'topic': 'black hole'})

result = model.invoke(prompt1)
print(result.content)
prompt2 = template2.invoke({'report': result.content})

result2 = model.invoke(prompt2)
print(result2.content)
