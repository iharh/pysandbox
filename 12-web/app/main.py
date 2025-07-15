import logging
import os
import sys
from typing import List, Optional, Union

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.data_structs.document_summary import IndexDocumentSummary
from llama_index.core.llms import LLM
from llama_index.core.schema import QueryType
from llama_index.core.storage.storage_context import DEFAULT_PERSIST_DIR
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.readers.web import SimpleWebPageReader
from qdrant_client import QdrantClient

from llama_index.core import Document, DocumentSummaryIndex, StorageContext, load_index_from_storage
from llama_index.node_parser.topic import TopicNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

SHOW_PROGRESS = True
LLM_MODEL_NAME = "llama3.2:3b"
COLLECTION_NAME = "summary_coll"
RECREATE_COLLECTION = True

URL_1 = "https://www.google.com"
# url="https://www.storynory.com/little-red-riding-hood-2"
# url="http://paulgraham.com/worked.html"
# url="https://en.wikipedia.org/wiki/Peter_Pan"
# url="https://edition.cnn.com/2025/07/14/politics/obama-democrats-message"

Q_1 = "What is Google?"
# q = "Who created Peter Pan?"

class RAGService:
    TOPIC_NODE_PARSER_SIMILARITY_METHOD="embedding" # | "llm"

    _llm: LLM
    _storage_context: StorageContext
    _document_summary_index: DocumentSummaryIndex
    _persist_dir: str

    def __init__(
            self,
            llm: LLM,
            embed_model: BaseEmbedding,
            vector_store: BasePydanticVectorStore,
            persist_dir: Optional[str] = None,
            recreate_collection: bool = False,
            show_progress: bool = False):

        self._llm = llm
        self._persist_dir = persist_dir
        self._storage_context = StorageContext.from_defaults(
            persist_dir=self._persist_dir,
            vector_store=vector_store
        )
        transformations = [
            TopicNodeParser.from_defaults(
                llm=self._llm,
                embed_model=embed_model,
                similarity_method=self.TOPIC_NODE_PARSER_SIMILARITY_METHOD,
                window_size=2,
            ),
        ]
        index_struct = IndexDocumentSummary()
        self._document_summary_index = DocumentSummaryIndex(
            llm=self._llm,
            embed_model=embed_model,
            index_struct=index_struct,
            transformations=transformations,
            embed_summaries=True,
            storage_context=self._storage_context,
            show_progress=show_progress,
        ) if recreate_collection else load_index_from_storage(
            self._storage_context,
            llm=self._llm,
            embed_model=embed_model,
        )

    def insert(self, documents: List[Document]):
        for document in documents:
            self._document_summary_index.insert(document=document)

    def persist(self, persist_dir: Union[str, os.PathLike]):
        self._storage_context.persist(persist_dir=persist_dir)

    def query(self, str_or_query_bundle: QueryType) -> RESPONSE_TYPE:
        summary_query_engine = self._document_summary_index.as_query_engine(
            llm=self._llm,
            response_mode="tree_summarize"
        )
        return summary_query_engine.query(str_or_query_bundle)

_logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    _logger.info("start")

    llm = Ollama(
        model=LLM_MODEL_NAME,
        request_timeout=120.0,
        context_window=8000,
    )  # pydantic_program_mode = PydanticProgramMode.LLM # ???
    embed_model = OllamaEmbedding(
        model_name=LLM_MODEL_NAME, # base_url="http://localhost:11434", ollama_additional_kwargs={"mirostat": 0},
    )
    qdrant_client = QdrantClient(host="localhost")  # port=6333,
    if RECREATE_COLLECTION and qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    qdrant_vector_store = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        client=qdrant_client,
    )

    svc = RAGService(
        llm=llm,
        embed_model=embed_model,
        vector_store=qdrant_vector_store,
        persist_dir=None if RECREATE_COLLECTION else DEFAULT_PERSIST_DIR,
        recreate_collection=RECREATE_COLLECTION,
        show_progress=SHOW_PROGRESS
    )

    reader = SimpleWebPageReader(html_to_text=True) # BeautifulSoupWebReader()

    documents = reader.load_data(urls=[URL_1])
    _logger.info(f"read {len(documents)} documents")

    if RECREATE_COLLECTION:
        svc.insert(documents),
        svc.persist(DEFAULT_PERSIST_DIR)
    _logger.info("done document summary indexing")

    response = svc.query(Q_1)
    _logger.info(response)

if __name__ == "__main__":
    main()