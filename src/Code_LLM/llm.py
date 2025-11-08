from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")

for chunk in llm.stream("Explain to me how to make carbonara pasta."):
    print(chunk.content, end="", flush=True)