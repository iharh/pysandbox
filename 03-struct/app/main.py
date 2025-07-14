from llama_index.core.llms.structured_llm import StructuredLLM
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core.llms import ChatResponse
from llama_index.core.bridge.pydantic import BaseModel

class Song(BaseModel):
    name: str
    artist: str

def main():
    print("start")
    llm = Ollama(
        model="llama3.2:3b",
        request_timeout=120.0,
        context_window=8000,
    )
    sllm: StructuredLLM = llm.as_structured_llm(Song)
    response: ChatResponse = sllm.chat([ChatMessage(role="user", content="Name a random song!")])

    print(response.message.content)

if __name__ == "__main__":
    main()