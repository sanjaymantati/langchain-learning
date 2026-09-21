from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


loader = WebBaseLoader(
    web_path="https://www.rfc-editor.org/info/rfc6749/"

)
docs = loader.load()


template = PromptTemplate(
    template="Write answer for the user query: {query} from the source webpage content: {data}",
    input_variables=['query', 'data']
)

parser = StrOutputParser()
model = ChatOpenAI(model = "gpt-5")

chain = template | model | parser

result = chain.invoke(
    {
        'query': "Is OAuth2.0 Stateless?",
        'data': docs
    }
)

print(result)

