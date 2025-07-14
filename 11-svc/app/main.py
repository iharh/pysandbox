import logging
import sys
from typing import Sequence

from llama_index.core.schema import BaseNode
from qdrant_client import QdrantClient

from llama_index.core import Document, DocumentSummaryIndex, Settings, StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.node_parser.topic import TopicNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

SHOW_PROGRESS = True
LLM_MODEL_NAME = "llama3.2:3b"
COLLECTION_NAME = "summary_coll"

Settings.embed_model = OllamaEmbedding(
    model_name=LLM_MODEL_NAME,
    base_url="http://localhost:11434",
    # ollama_additional_kwargs={"mirostat": 0},
)
Settings.llm = Ollama(
    model=LLM_MODEL_NAME,
    request_timeout=120.0,
    context_window=8000,
) # pydantic_program_mode = PydanticProgramMode.LLM # ???

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

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    print("start")

    client = QdrantClient(host="localhost") # port=6333,

    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)

    qdrant_vector_store = QdrantVectorStore(
        collection_name=COLLECTION_NAME,
        client=client,
    )

    kw_sum_transformations = [ # TitleExtractor(nodes=5),
        TopicNodeParser.from_defaults(
            # tokenizer=TokenTextSplitter(chunk_size=512, chunk_overlap=128),
            similarity_method="embedding", # ? "llm"
            window_size=2,
        ),
        # SummaryExtractor(),  # metadata["section_summary"]
    ]

    pipeline = IngestionPipeline(
        transformations=kw_sum_transformations,
        vector_store=qdrant_vector_store,
    )

    doc1 = Document(text=CONTEXT_TEXT_1)
    doc2 = Document(text=CONTEXT_TEXT_2)
    my_docs = [doc1, doc2]
    extracted_nodes: Sequence[BaseNode] = pipeline.run(
        documents=my_docs,
        show_progress=SHOW_PROGRESS,
    )

    print(f"done running pipeline. nodes: {len(extracted_nodes)}")

    storage_context_summary = StorageContext.from_defaults(vector_store=qdrant_vector_store)
    #storage_context_summary.docstore.add_documents(extracted_nodes)

    # just save to qdrant
    document_summary_index = DocumentSummaryIndex(
        nodes = extracted_nodes,
        embed_summaries=True,
        storage_context=storage_context_summary,
        show_progress=SHOW_PROGRESS,
    )

    print("done document summary indexing")

    summary_query_engine = document_summary_index.as_query_engine(response_mode="tree_summarize")
    response = summary_query_engine.query("Who created Peter Pan?")

    #summary_query_retriever = document_summary_index.as_retriever() # response_mode="tree_summarize"
    #response = summary_query_retriever.retrieve("Who created Peter Pan?")

    print(response)

if __name__ == "__main__":
    main()