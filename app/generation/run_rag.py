from app.generation.rag import RAGPipeline


rag = RAGPipeline()

question = "How does Requests make a GET request?"

answer = rag.ask(question)

print("\nANSWER:\n")
print(answer)