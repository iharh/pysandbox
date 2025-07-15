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

# ? SentenceSplitter
# ? RetrieverTool
# ? TitleExtractor
# ? Zephyr query engine
# ? pip install llama-index-embeddings-huggingface
# from zephyr_pack.base import ZephyrQueryEnginePack
# llamaindex-cli cmd-line tool, rag command
# pip install html2text
```
## indices

* core/indices/registry.py

```python
INDEX_STRUCT_TYPE_TO_INDEX_CLASS: Dict[IndexStructType, Type[BaseIndex]] = {
    IndexStructType.TREE: TreeIndex,
    IndexStructType.LIST: SummaryIndex,
    IndexStructType.KEYWORD_TABLE: KeywordTableIndex,
    IndexStructType.VECTOR_STORE: VectorStoreIndex,
    IndexStructType.SQL: SQLStructStoreIndex,
    IndexStructType.PANDAS: PandasIndex,  # type: ignore
    IndexStructType.KG: KnowledgeGraphIndex,
    IndexStructType.SIMPLE_LPG: PropertyGraphIndex,
    IndexStructType.EMPTY: EmptyIndex,
    IndexStructType.DOCUMENT_SUMMARY: DocumentSummaryIndex,
    IndexStructType.MULTIMODAL_VECTOR_STORE: MultiModalVectorStoreIndex,
}
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

links:
* https://docs.llamaindex.ai/en/stable/understanding/extraction/structured_prediction/

## query

links:
* https://docs.llamaindex.ai/en/stable/module_guides/querying/
* https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/
* https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/usage_pattern/
* https://docs.llamaindex.ai/en/stable/examples/query_engine/custom_query_engine/
* https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers/
* https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/
* https://docs.llamaindex.ai/en/stable/examples/usecases/10k_sub_question/
* https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/RetrieverRouterQueryEngine.ipynb#scrollTo=w7wpPTy_F8Z9

query pipeline:
* https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline/

summarizer:
* summarizer = TreeSummarize(llm=llm)

other:
* https://docs.llamaindex.ai/en/stable/examples/vector_stores/qdrant_bm42/
* https://github.com/qdrant/examples/blob/949669f001a03131afebf2ecd1e0ce63cab01c81/llama_index_recency/Qdrant%20and%20LlamaIndex%20%E2%80%94%20A%20new%20way%20to%20keep%20your%20Q%26A%20systems%20up-to-date.ipynb

## readers-web

## book

to-look:
* p64 llama-inex-readers-web, html2text -> SimpleWebPageReader
* p75 HTMLNodeParser (uses BeautifulSoup) ??? split nodes based on tags - https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/html/
* p82 metadata extractors (check below)
* p83 llama_index.core.extractors.SummaryExtractor(summaries=[...]) ... and the actual prompt in prompt_template parameter
* p84 TitleExtractor
* p84-85 EntityExtractor (NLTK NER), DEFAULT_ENTITY_MODEL = "tomaarsen/span-marker-mbert-base-multinerd" (uv add span-marker)
* https://docs.llamaindex.ai/en/stable/api_reference/extractors/entity/
* p86 KeywordExtractor(keywords=3)
* https://docs.llamaindex.ai/en/stable/api_reference/extractors/keyword/
* p96 IngestionPipeline (aggregate multiple ...), Settings.transformations at p97
* p101 ...
* response = query_engine.query("What are the main topics covered in the dataset?") - https://www.cohorte.co/blog/mastering-dataset-indexing-with-llamaindex-a-complete-guide
* p115 StorageContext, 
* index = VectorStoreIndex.from_vector_store(vector_store=qdrant_vector_store)
* p118 SummaryIndex
* p120 DocumentSummaryIndex
* p122 KeywordTableIndex, keyword-to-node mapping, index = KeywordTableIndex.from_documents(docs); query_engine = index.as_query_engine();

## API

query pipeline:
* https://docs.llamaindex.ai/en/stable/api_reference/query_pipeline/

## Indices

TODO:
* insert_nodes(...)

important diagram:
* https://medium.com/@aneesha161994/question-answering-in-rag-using-llama-index-92cfc0b4dae3

```python

# vector_store/base.py
class VectorStoreIndex(BaseIndex[IndexDict]):
    # ...
    def _add_nodes_to_index(
        self,
        index_struct: IndexDict,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **insert_kwargs: Any,
    ) -> None:
        """Add document to index."""
        if not nodes:
            return

        for nodes_batch in iter_batch(nodes, self._insert_batch_size):
            nodes_batch = self._get_node_with_embedding(nodes_batch, show_progress)
            new_ids = self._vector_store.add(nodes_batch, **insert_kwargs)
            # ...
```

## Topic identification

* TopicNodeParser
* https://docs.llamaindex.ai/en/stable/api_reference/node_parser/topic/
* !!! https://docs.llamaindex.ai/en/stable/examples/node_parsers/topic_parser/
* node split based on topics


## Retrievers

composable, ensemble:
* IndexNode, SummaryIndex (!!! modes)
* https://docs.llamaindex.ai/en/stable/examples/retrievers/composable_retrievers/
* https://docs.llamaindex.ai/en/stable/examples/retrievers/ensemble_retrieval/

to-look:
* QueryFusionRetriever(mode: FUSION_MODES = FUSION_MODES.SIMPLE, ...), fusion_retriever.py
* https://docs.llamaindex.ai/en/stable/examples/retrievers/simple_fusion/
* AutoMergingRetriever
* https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/
* RouterRetriever (PydanticMultiSelector)
* https://docs.llamaindex.ai/en/stable/examples/retrievers/router_retriever/
* https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_vs_recursive_retriever/

for indices:
* KeywordTableIndex, KeywordTableGPTRetriever, 
*   utils.py -> extract_keywords_given_response <start_token>: <word1>, <word2>, ...
* DocumentSummaryIndex, DocumentSummaryIndexLLMRetriever, DocumentSummaryIndexEmbeddingRetriever

samples:
* https://docs.llamaindex.ai/en/stable/examples/cookbooks/oreilly_course_cookbooks/Module-6/Router_And_SubQuestion_QueryEngine/

## other

look:
* external-libs/site-packages/llama_index/core/...
* prompt_type.py
* default_prompts.py - find src

```python
# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.callback_manager = callback_manager

######
import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, StorageContext, load_index_from_storage

# Load documents from a directory
documents = SimpleDirectoryReader("data").load_data()

# Add metadata dynamically from file name and page number
documents_with_metadata = []
for doc in documents:
    file_name = os.path.basename(doc.metadata["file_path"])
    page_number = doc.metadata.get("page_number", "unknown")
    doc.metadata.update({"source": file_name, "page": page_number})
    documents_with_metadata.append(doc)

# Create a vector store and persist locally
storage_context = StorageContext.from_defaults(persist_dir="local_index_store")
index = GPTVectorStoreIndex.from_documents(docs=documents_with_metadata, storage_context=storage_context)
index.storage_context.persist("local_index_store")

# Load the index from local storage
index = load_index_from_storage(StorageContext.from_defaults(persist_dir="local_index_store"))

# Query with metadata filtering
query_engine = index.as_query_engine()
response = query_engine.query("Find insights from the financial report")
print(response)

```
* https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/ensemble_query_engine.ipynb
* PydanticMultiSelector
* https://docs.llamaindex.ai/en/stable/examples/retrievers/router_retriever/

* https://www.llamaindex.ai/blog/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec
* https://docs.llamaindex.ai/en/stable/examples/index_structs/doc_summary/DocSummary/

retrieval models:
* llm
* embeddings

## web

links:
* https://docs.llamaindex.ai/en/stable/examples/data_connectors/WebPageDemo/

* curl 'http://localhost:5000?q=google'
