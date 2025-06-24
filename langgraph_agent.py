import os
import json
from typing import Annotated, AsyncGenerator
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
# from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

# Obtém a chave da API do ambiente
api_key = os.getenv("OPENAI_API_KEY")

class State(TypedDict):
    question: Annotated[str, "Question"]
    messages: Annotated[list, add_messages]

# llm = ChatOpenAI(model="claude-3-haiku-20240307", anthropic_api_key=api_key)
# llm = ChatOpenAI(model="gpt-4o-mini")      
ollama_url = "http://192.168.50.70:11434"
model_name = "gemma3:12b"
llm = ChatOllama(
                # base_url='http://host.docker.internal:11434',
                base_url=ollama_url,
                model=model_name,
                temperature=0,
                context_window=8192,
            )  

def chatbot(state: State):
    print(f'[langgraph_agent.py] question: {state["question"]}')
    print(f'[langgraph_agent.py] messages: {state["messages"]}')
    return {"messages": [llm.invoke(state["question"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# ----
# import os
# import json
# from typing import Annotated, AsyncGenerator
# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages
# # from langchain_anthropic import ChatAnthropic
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# # Carrega as variáveis do arquivo .env
# load_dotenv()

# # Obtém a chave da API do ambiente
# api_key = os.getenv("OPENAI_API_KEY")

# class State(TypedDict):
#     question: Annotated[str, "Question"]  # 질문(누적되는 list)
#     context: Annotated[str, "Context"]  # 문서의 검색 결과
#     answer: Annotated[str, "Answer"]  # 답변
#     messages: Annotated[list, add_messages]  # 메시지(누적되는 list)
#     relevance: Annotated[str, "Relevance"]  # 관련성

# llm = ChatOpenAI(model="gpt-4o-mini")

# def chatbot(state: State):
#     print(f"[langgraph_agent.py] question : {state['question']}")        
#     print(f"[langgraph_agent.py] messages : {state['messages']}")        
#     print("--" * 40)
#     return {"messages": [llm.invoke(state["question"])]}
#     # return {"messages": [llm.invoke(state["question"])]}

# graph_builder = StateGraph(State)
# graph_builder.add_node("chatbot", chatbot)
# graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)

# graph = graph_builder.compile()
