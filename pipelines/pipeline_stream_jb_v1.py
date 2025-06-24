import requests
import json
from typing import List, Union, Generator, Iterator
try:
    from pydantic.v1 import BaseModel
except Exception:
    from pydantic import BaseModel



class Pipeline:

    class Valves(BaseModel):
        pass

    def __init__(self):
        self.id = "LangGraph Agent (Stream JB)"
        self.name = "척척박사 (Stream)"

    async def on_startup(self):
        print(f"on_startup: {__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
            ) -> Union[str, Generator, Iterator]:
        url = 'http://host.docker.internal:8001/openwebui-pipelines/api/stream'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        data = {
            "question": user_message,
            "context": "",
            "answer": "",
            "messages": messages,
            "relevance": ""
            }

        # print(f"[{self.id}] data: {data}")
        # print(f"[{self.id}] model_id: {model_id}")
        # print(f"[{self.id}] body: {body}")
        
        response = requests.post(url, json=data, headers=headers, stream=True)
        
        response.raise_for_status()
        
        return response.iter_lines()