import logging
import sys
from typing import List

from llama_index.core.data_structs.document_summary import IndexDocumentSummary
from llama_index.core.llms import LLM
from llama_index.core.llms.utils import LLMType
from llama_index.core.schema import TransformComponent
from qdrant_client import QdrantClient

from llama_index.core import Document, DocumentSummaryIndex, Settings, StorageContext
from llama_index.node_parser.topic import TopicNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

SHOW_PROGRESS = True
LLM_MODEL_NAME = "llama3.2:3b"

Settings.embed_model = OllamaEmbedding(
    model_name=LLM_MODEL_NAME,
    base_url="http://localhost:11434", # ollama_additional_kwargs={"mirostat": 0},
)

CONTEXT_TEXT_1 = """
Peter Pan is a fictional character created by Scottish novelist and playwright J. M. Barrie. 
A free-spirited and mischievous young boy who can fly and never grows up, Peter Pan spends his never-ending childhood having adventures on the mythical island of Neverland as the leader of the Lost Boys, interacting with fairies, pirates, mermaids, Native Americans, and occasionally ordinary children from the world outside Neverland.

Peter Pan has become a cultural icon symbolizing youthful innocence and escapism. 
In addition to two distinct works by Barrie, The Little White Bird (1902, with chapters 13–18 published in Peter Pan in Kensington Gardens in 1906), and the West End stage play Peter Pan; or, the Boy Who Wouldn't Grow Up (1904, which expanded into the 1911 novel Peter and Wendy), the character has been featured in a variety of media and merchandise, both adapting and expanding on Barrie's works. 
These include several films, television series and many other works.
"""

CONTEXT_TEXT_2 = """
Once upon a time there lived a little country girl, the prettiest creature who was ever seen. Her mother had a little red riding hood made for her.
Everybody called her Little Red Riding Hood.

One day her mother said to her: “Go my dear, and see how your grandmother is doing, for I hear she has been very ill.”

Little Red Riding Hood set out immediately.

As she was going through the wood, she met with a wolf. He asked her where she was going.
"""

class RAGService:
    COLLECTION_NAME = "summary_coll"
    TOPIC_NODE_PARSER_SIMILARITY_METHOD="embedding" # | "llm"

    _llm: LLM
    _qdrant_client: QdrantClient
    _qdrant_vector_store: QdrantVectorStore
    _storage_context: StorageContext
    _kw_sum_transformations: List[TransformComponent]
    _index_struct: IndexDocumentSummary
    _document_summary_index: DocumentSummaryIndex
    _show_progress: bool

    def __init__(
            self,
            llm: LLMType,
            qdrant_client: QdrantClient,
            show_progress: bool):

        self._llm = llm
        self._qdrant_client = qdrant_client
        self._show_progress = show_progress
        self._qdrant_vector_store = QdrantVectorStore(
            collection_name=self.COLLECTION_NAME,
            client=self._qdrant_client,
        )
        self._storage_context = StorageContext.from_defaults(vector_store=self._qdrant_vector_store)
        self._kw_sum_transformations = [
            TopicNodeParser.from_defaults(
                llm=self._llm,
                similarity_method=self.TOPIC_NODE_PARSER_SIMILARITY_METHOD,
                window_size=2,
            ),
        ]
        self._index_struct = IndexDocumentSummary()
        self._document_summary_index = DocumentSummaryIndex(
            llm=self._llm,
            index_struct=self._index_struct,
            transformations=self._kw_sum_transformations,
            embed_summaries=True,
            storage_context=self._storage_context,
            show_progress=self._show_progress,
        )

    def recreate_collection(self):
        if self._qdrant_client.collection_exists(collection_name=self.COLLECTION_NAME):
            self._qdrant_client.delete_collection(collection_name=self.COLLECTION_NAME)
        self._qdrant_client.create_collection(collection_name=self.COLLECTION_NAME)

    def insert(
            self,
            document: Document):
        self._document_summary_index.insert(document=document)


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    print("start")

    llm = Ollama(
        model=LLM_MODEL_NAME,
        request_timeout=120.0,
        context_window=8000,
    )  # pydantic_program_mode = PydanticProgramMode.LLM # ???

    qdrant_client = QdrantClient(host="localhost")  # port=6333,

    svc = RAGService(
        llm=llm,
        qdrant_client=qdrant_client,
        show_progress=SHOW_PROGRESS
    )
    svc.recreate_collection()
    svc.insert(Document(text=CONTEXT_TEXT_1))
    svc.insert(Document(text=CONTEXT_TEXT_2))
    print("done document summary indexing")

    #summary_query_engine = document_summary_index.as_query_engine(response_mode="tree_summarize")
    #response = summary_query_engine.query("Who created Peter Pan?")
    #print(response)

if __name__ == "__main__":
    main()