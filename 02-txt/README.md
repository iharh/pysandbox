# ollama

Shell stuff:
```shell
ollama run llama2 --input "Summarize the following text: some text"
ollama pull [model-name]
```

Python Sample:
```python
import subprocess

def run_ollama(model, prompt):
    result = subprocess.run(
        ["ollama", "run", model, "--input", prompt],
        stdout=subprocess.PIPE,
        text=True
    )
    return result.stdout

# Summarization
text_to_summarize = "Ollama is an AI-based tool for running large language models locally on your device or in the cloud. It is designed to be extensible and customizable, enabling a variety of use cases. The tool aims to provide efficient solutions for tasks like text summarization and topic identification."
summary = run_ollama("llama2", f"Summarize the following text: {text_to_summarize}")
print("Summary:", summary)

# Topic Identification
topics = run_ollama("llama2", f"What are the main topics or keywords in the following text? {text_to_summarize}")
print("Topics:", topics)
```


# free models

## ollama

* https://docs.llamaindex.ai/en/stable/examples/llm/ollama/
* https://docs.llamaindex.ai/en/stable/examples/llm/ollama/#thinking
* https://ollama.com/blog/structured-outputs

## BentoML OpenLLM

links:
* https://docs.llamaindex.ai/en/stable/examples/llm/openllm/

## Perplexity

links:
* https://docs.llamaindex.ai/en/stable/examples/llm/perplexity/


# Summarization

sample prompts:
* Provide a one-sentence summary...
* List three main topics...

# Langchain

TBD

# LlamaIndex

resources:

* https://www.llamaindex.ai
* https://www.llamaindex.ai/blog
* https://www.llamaindex.ai/community
* https://www.llamaindex.ai/llamaparse
* https://docs.llamaindex.ai/en/stable/getting_started/starter_tools/rag_cli/
* https://llamahub.ai/
* https://www.reddit.com/r/LlamaIndex/
* https://www.youtube.com/@LlamaIndex
* https://github.com/run-llama

llamadeploy:
* https://docs.llamaindex.ai/en/stable/module_guides/llama_deploy/
* https://github.com/run-llama/llama_deploy

extract:
* https://www.llamaindex.ai/llamaextract

packages:
* https://pypi.org/project/llama-index/
* https://pypi.org/project/llama-index-core/
* https://github.com/run-llama/llama_index
* https://www.npmjs.com/package/create-llama

docs:
* https://github.com/run-llama/llama_index
* https://docs.llamaindex.ai/en/latest/
* https://docs.llamaindex.ai/en/stable/
* https://docs.llamaindex.ai/en/stable/getting_started/concepts/

models:
* https://docs.llamaindex.ai/en/stable/module_guides/models/
* https://docs.llamaindex.ai/en/stable/module_guides/models/llms/
* https://docs.llamaindex.ai/en/stable/module_guides/models/llms/modules/
* https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/index.md
* https://github.com/run-llama/llama_index/tree/main/docs/docs/examples/llm

sample ollama:
* https://docs.llamaindex.ai/en/stable/examples/llm/ollama/
* https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/llm/ollama.ipynb
* https://blog.google/technology/developers/gemma-open-models/
* https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/llm/ollama_gemma.ipynb
* pip install llama-index-llms-ollama
* from llama_index.llms.ollama import Ollama

prompting:
* https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/
* https://docs.llamaindex.ai/en/stable/module_guides/models/prompts/usage_pattern/

indexing:
* https://docs.llamaindex.ai/en/stable/module_guides/indexing/

query engine:
* https://docs.llamaindex.ai/en/stable/module_guides/querying/
* https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/
* https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/
* https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/usage_pattern/

learn:
* https://docs.llamaindex.ai/en/stable/understanding/
* https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms/
* https://docs.llamaindex.ai/en/stable/understanding/rag/

use cases:
* https://docs.llamaindex.ai/en/stable/use_cases/extraction/

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

## samples

links:
* https://github.com/run-llama/llamabot

# Semantic Kernel

TBD:

# Models hosting

## Replicate

links:
* https://replicate.com/
* https://replicate.com/explore
* https://replicate.com/pricing
