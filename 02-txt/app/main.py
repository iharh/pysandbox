from llama_index.llms.ollama import Ollama

def main():
    print("start")
    llm = Ollama(
        model="llama3.2:3b",
        request_timeout=120.0,
        # Manually set the context window to limit memory usage
        context_window=8000,
    )
    resp = llm.complete("Who is Paul Graham?")
    print(resp)

if __name__ == "__main__":
    main()
