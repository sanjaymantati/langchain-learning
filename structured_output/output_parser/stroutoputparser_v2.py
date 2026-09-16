from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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



parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({'topic': 'black hole'})
print(result)
