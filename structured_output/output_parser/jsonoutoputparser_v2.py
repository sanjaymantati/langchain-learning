from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv
import os
load_dotenv()

#ENDPOINT VERSION
# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

# LOCAL VERSION
# import torch
#
# print("GPU:", torch.cuda.get_device_name(0))
#
#
# os.environ['HF_HOME'] = "D:/models/hf"
# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     device=0,
#     pipeline_kwargs=dict(temperature=0.1,
#                          max_new_tokens=100)
# )
# model = ChatHuggingFace(llm=llm)

# OpenAI

model = ChatOpenAI()

parser = JsonOutputParser()
template = PromptTemplate(template="give me the name, age and city of a fictional person \n {format_instruction}",
                          input_variables=[],
                          partial_variables={'format_instruction':parser.get_format_instructions()})
prompt = template.format()

print(prompt)

chain = template | model | parser
result = chain.invoke({})
print(result)