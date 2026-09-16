from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



## chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful assistant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history= []
## Load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines()) 

print(chat_history)

## Create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': 'What is the value of multiplication we just did?'})

print(prompt)
