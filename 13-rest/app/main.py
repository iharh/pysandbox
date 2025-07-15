import logging
import os
import sys
from typing import List, Optional, Union, Annotated

import uvicorn
from fastapi import FastAPI, APIRouter, Depends, Query
from fastapi_utils.api_model import APIModel
from fastapi_utils.cbv import cbv
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.data_structs.document_summary import IndexDocumentSummary
from llama_index.core.llms import LLM
from llama_index.core.schema import QueryType
from llama_index.core.storage.storage_context import DEFAULT_PERSIST_DIR
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.readers.web import SimpleWebPageReader
from pydantic import BaseModel
from qdrant_client import QdrantClient

from llama_index.core import Document, DocumentSummaryIndex, StorageContext, load_index_from_storage
from llama_index.node_parser.topic import TopicNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

SHOW_PROGRESS = True
LLM_MODEL_NAME = "llama3.2:3b"
COLLECTION_NAME = "summary_coll"

class RAGService(BaseModel):
    TOPIC_NODE_PARSER_SIMILARITY_METHOD: str = "embedding" # | "llm"

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

        super().__init__()
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

router = APIRouter()

def make_rag_service() -> RAGService:
    llm = Ollama(
        model=LLM_MODEL_NAME,
        request_timeout=120.0,
        context_window=8000,
    )  # pydantic_program_mode = PydanticProgramMode.LLM # ???
    embed_model = OllamaEmbedding(
        model_name=LLM_MODEL_NAME, # base_url="http://localhost:11434", ollama_additional_kwargs={"mirostat": 0},
    )
    qdrant_client = QdrantClient(host="localhost")  # port=6333,
    qdrant_vector_store = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        client=qdrant_client,
    )
    return RAGService(
        llm=llm,
        embed_model=embed_model,
        vector_store=qdrant_vector_store,
        persist_dir=DEFAULT_PERSIST_DIR,
        recreate_collection=False,
        show_progress=SHOW_PROGRESS
    )

RAGServiceDep = Annotated[RAGService, Depends(make_rag_service)]

class GeneralInfo(APIModel):
    response: str

class IndexItem(BaseModel):
    url: str

@cbv(router)
class RAGController:
    @router.post("/", response_model=GeneralInfo)
    def insert(self, rag_service: RAGServiceDep, index_item: IndexItem):
        reader = SimpleWebPageReader(html_to_text=True) # BeautifulSoupWebReader()
        documents = reader.load_data(urls=[index_item.url])
        documents_count = len(documents)
        _logger.info(f"read {documents_count} documents")
        rag_service.insert(documents),
        rag_service.persist(DEFAULT_PERSIST_DIR)
        _logger.info("done documents persisting")
        return GeneralInfo(response=f"done persisting {documents_count} documents")

    @router.get("/", response_model=GeneralInfo)
    def query(self, rag_service: RAGServiceDep, q: str = Query()):
        response = rag_service.query(q)
        _logger.info(f"request: {q}, response: {response}")
        return GeneralInfo(response=response.response)

app = FastAPI(debug=True)
app.include_router(router)

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    uvicorn.run(app, port = 5000)

if __name__ == "__main__":
    main()