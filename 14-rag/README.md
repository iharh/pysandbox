# RAG sample

Task Description:

    1. News Extraction: 
        1.1. Develop a script to scrape news articles from provided URLs.
        1.2. Ensure the extracted content captures the full text and headline of the articles.
    2. GenAI-driven Summarization and Topic Identification:
        2.1. Use a GenAI platform or tool (e.g. OpenAI's GPT models, or any other LLM) to analyze the articles. 
             Your tasks will include generating a summary that captures key points and identifying the main topics of each article.
        2.2. The focus should be on effectively integrating and utilizing GenAI tools rather than building from scratch.
    3. Semantic Search with GenAI:
        3.1. Store the extracted news, their GenAI-generated summaries, and topics in a vector database.
        3.2. Implement a semantic search feature leveraging GenAI tools to interpret and find relevant articles based on user queries. 
             This search should understand the context of the queries and match them effectively with the summaries and topics. 
             Search should handle semantically close search terms like synonyms.

## prerequisites

In order to start the sample, please ensure the following has been installed:
* docker
* docker compose v2
* uv

## starting 3-rd party services

Use the following command to start 3-rd party services:
* docker compose up -d
* docker exec -it ollama ollama pull llama3.2:3b
* docker exec -it ollama ollama list

## running 

Use the following command to run the RAG sample:
* uv run app/main.py

## checking

To index some sample document(article) from a given url, run:
* sh/idx.sh

To query for some sample query text, run:
* sh/q.sh
