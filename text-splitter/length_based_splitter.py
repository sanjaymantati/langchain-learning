from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TextSplitter, CharacterTextSplitter


text = None
with open('solar-system.txt') as f:
    text= f.read()


loader =PyPDFLoader(
    file_path="Indian_Constitution_Drafting_Committee_Report.pdf"
)
docs = loader.load()
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=' ',
)

result  = splitter.split_documents(docs)

print(result)