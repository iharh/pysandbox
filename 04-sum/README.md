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
* https://blog.google/technology/developers/gemma-open-models/
* https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/llm/ollama_gemma.ipynb
* pip install llama-index-llms-ollama
* from llama_index.llms.ollama import Ollama

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
