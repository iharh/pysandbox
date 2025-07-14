import logging
import sys

from llama_index.embeddings.ollama import OllamaEmbedding

def main():
    print("start")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    ollama_embedding = OllamaEmbedding(
        model_name="llama3.2:3b",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0},
    )

    pass_embedding = ollama_embedding.get_text_embedding_batch(
        ["This is a passage!", "This is another passage"],
        show_progress=True
    )
    print(pass_embedding)

if __name__ == "__main__":
    main()