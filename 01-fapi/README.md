# title

Sample code
```
# app.py
# https://www.youtube.com/watch?v=1XT4NDh1xnc

class GuidedConversationApp(ChatCompletion):
    def __init__(self):
        self.__survey = parse_yaml_file("resources/tourist_survery.yaml")
    
    async def chat_completion(self, request: Request, response: Response) -> None:
        chain = create_chain()
        with ...

app = DIALApp(dial_url=config.DIAL_URL, propagation_auto_headers=True, add_healthcheck=True)
app.add_chat_completion("guided-conversation", GuidedConversationApp())

# Run built app
if __name__ == "__main__":
    uvicorn.run(app, port=5000)

# GET http://127.0.0.1:5000/docs
# GET http://127.0.0.1:5000/redoc

# GET http://127.0.0.1:5000/openapi.json

# ???
# GET http://127.0.0.1:5000/admin

# bash suggests to install "fastapi-cli-slim"
# "fastapi", "run", "--workers", "4", "app/main.py"

```

https://fastapi.tiangolo.com/
https://fastapi.tiangolo.com/deployment/manually
