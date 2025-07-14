# ollama

# free models

## ollama

* https://docs.llamaindex.ai/en/stable/examples/llm/ollama/
* https://ollama.com/blog/structured-outputs

# Summarization

sample prompts:
* Provide a one-sentence summary...
* List three main topics...

# LlamaIndex

sample ollama:
* https://docs.llamaindex.ai/en/stable/examples/llm/ollama/
* https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/llm/ollama.ipynb

loading documents and nodes:
* https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/
* https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents/
* https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_nodes/
* https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_metadata_extractor/

loading data connectors:
* https://docs.llamaindex.ai/en/stable/module_guides/loading/connector/
* https://docs.llamaindex.ai/en/stable/module_guides/loading/connector/modules/

loading node parsers and text splitters:
* https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
* https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/
* https://github.com/grantjenks/py-tree-sitter-languages
* https://github.com/tree-sitter/tree-sitter-html

loading ingestion pipeline
* https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/
* https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/transformations/

indexing:
* https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index

storing:
* https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/
* https://docs.llamaindex.ai/en/stable/api_reference/storage/index_store/postgres/
* https://docs.llamaindex.ai/en/stable/examples/docstore/MongoDocstoreDemo/
* https://docs.llamaindex.ai/en/stable/examples/docstore/CloudSQLPgDocstoreDemo/
* https://docs.llamaindex.ai/en/stable/examples/docstore/RedisDocstoreIndexStoreDemo/
* https://llamahub.ai/?tab=storage

postgres:
* https://llamahub.ai/l/storage/llama-index-storage-index-store-postgres
* https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/
* https://github.com/pgvector/pgvector
* https://github.com/pgvector/pgvector#docker

postgresml:
* https://llamahub.ai/l/indices/llama-index-indices-managed-postgresml
* https://github.com/postgresml/postgresml



```python
from llama_index.vector_stores.pinecone import PineconeVectorStore
import logging

logging.basicConfig(level=logging.DEBUG)


from llama_index.core import Settings, Document, VectorStoreIndex
doc = Document(
    text="Sentence 1. Sentence 2. Sentence 3."
)

# p35 - manual docs
# p40 - auto-extract nodes with splitters 
# p44 - good overview of arch ...
# p47 - !!! diagram fig 3.5
# ch4 from p62

# https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings/

# ? SimpleDirectoryReader
# ? SentenceSplitter
# ? SummaryIndex
# ? DocumentSummaryIndex
# ? RetrieverTool
# ? TitleExtractor
# ? Zephyr query engine
# ? pip install llama-index-embeddings-huggingface
# from zephyr_pack.base import ZephyrQueryEnginePack
# llamaindex-cli cmd-line tool, rag command
# pip install html2text

```

## qdrant

links:
* https://docs.llamaindex.ai/en/stable/api_reference/readers/qdrant/
* https://docs.llamaindex.ai/en/stable/api_reference/storage/vector_store/qdrant/
* https://qdrant.tech/documentation/beginner-tutorials/neural-search/
* https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/
* https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/

## embeddings

links:
* https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/
* https://docs.llamaindex.ai/en/stable/examples/embeddings/ollama_embedding/

* https://docs.llamaindex.ai/en/stable/examples/embeddings/Langchain/
* https://python.langchain.com/docs/integrations/text_embedding/
* https://python.langchain.com/api_reference/ollama/embeddings/langchain_ollama.embeddings.OllamaEmbeddings.html

## misc

learn:
* https://docs.llamaindex.ai/en/stable/understanding/
* https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms/
* https://docs.llamaindex.ai/en/stable/understanding/rag/

use cases:
* https://docs.llamaindex.ai/en/stable/use_cases/extraction/
* https://docs.llamaindex.ai/en/stable/examples/structured_outputs/structured_outputs/

notes:
* cloud have a free plan (10000 creds/mo)
* check https://github.com/run-llama/llama_index/tree/main/llama-index-integrations
* check https://docs.llamaindex.ai/en/stable/examples/vector_stores/SimpleIndexDemoLlama-Local/

## good overview

links:
* https://docs.llamaindex.ai/en/stable/understanding/rag/
* https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/index.md
* https://docs.llamaindex.ai/en/stable/getting_started/starter_tools/rag_cli/
* https://docs.llamaindex.ai/en/stable/examples/llm/ollama/

## prompt templates

```python
from llama_index.llms.groq import Groq
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel

class Restaurant(BaseModel):
    name: str
    city: str
    cuisine: str

llm = Groq(model="llama3-8b-8192", pydantic_program_mode="llm")
prompt_tmpl = PromptTemplate(
    "Generate a restaurant in a given city {city_name}"
)
restaurant_obj = llm.structured_predict(
    Restaurant, prompt_tmpl, city_name="Miami"
)
print(restaurant_obj)
```

```python

# llm.py
    @dispatcher.span
    def predict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[List[ChatMessage]] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> "AgentChatResponse":
...
        task = Task(
            input=user_msg,
            memory=ChatMemoryBuffer.from_defaults(chat_history=chat_history),
            extra_state={},
            callback_manager=self.callback_manager,
        )


# function_program.py

_allow_parallel_tool_calls


# program/utils.py
def get_program_for_llm(
    output_cls: Type[Model],
    prompt: BasePromptTemplate,
    llm: LLM,
    pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
    **kwargs: Any,
) -> BasePydanticProgram[Model]:



```

## struct

* https://docs.llamaindex.ai/en/stable/understanding/extraction/structured_prediction/
