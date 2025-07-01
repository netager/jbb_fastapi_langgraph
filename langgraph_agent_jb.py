import os
import json
from typing import List, Annotated, AsyncGenerator
from typing_extensions import TypedDict
from urllib.parse import urlencode

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_teddynote.messages import messages_to_history
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_teddynote.evaluator import GroundednessChecker
from langgraph.checkpoint.memory import MemorySaver

from rag.utils import format_docs
from rag.pdf import PDFRetrievalChain

from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith("CH17-LangGraph-Structures")

# PDF 문서를 로드합니다.
# pdf = PDFRetrievalChain(["data/SPRI_AI_Brief_2023년12월호_F.pdf"]).create_chain()
pdf = PDFRetrievalChain().create_chain()

# retriever와 chain을 생성합니다.
pdf_retriever = pdf.retriever
pdf_chain = pdf.chain

# # Obtém a chave da API do ambiente
# api_key = os.getenv("OPENAI_API_KEY")

class State(TypedDict):
    question: Annotated[str, "Question"]  # 질문(누적되는 list)
    context: Annotated[list, "Context"]  # 문서의 검색 결과
    answer: Annotated[str, "Answer"]  # 답변
    messages: Annotated[list, add_messages]  # 메시지(누적되는 list)
    relevance: Annotated[str, "Relevance"]  # 관련성


def print_state(state, func_name) -> None:
    print(f"- [{func_name}]-------------------------------------")
    print(f"question: {state['question']}")
    # print(f"context: {type(state['context'])}, {len(state['context'])}, {state['context']}")
    for context in state['context']:
        # print(f"context: {context['page_context']}")
        print(f"context: {context.metadata['source']}")

    # print(f"answer: {state['answer'][:20]}")
    print(f"messages: {state['messages'][-1]}")
    print(f"relevance: {state['relevance']}")
    print("--------------------------------------")

# 문서 검색 노드
def retrieve_document(state: State) -> State:
    
    print_state(state, 'retrieve_document')

    # print(f"[langgraph_agent_jb.py] state : {state['messages']}")        

    # 질문을 상태에서 가져옵니다.
    latest_question = state["question"]

    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
    retrieved_docs = pdf_retriever.invoke(latest_question)
    # print(f'[langgraph_agent_jp][retrieve_document] {latest_question}')
    # print('-' * 60)
    # print(f'[langgraph_agent_jp][retrieve_document] {retrieved_docs}')   # TODO: Delete

    # 검색된 문서를 형식화합니다.(프롬프트 입력으로 넣어주기 위함)
    # retrieved_docs = format_docs(retrieved_docs)

    # 검색된 문서를 context 키에 저장합니다.
    return State(context=retrieved_docs)

        
def llm_answer(state: State) -> State:
    print_state(state, 'llm_answer')

    latest_question = state["question"]
    context = format_docs(state["context"])
    # context = state["context"]

    # print(f'[langgraph_agent_jp][llm_answer()] <{latest_question}>{type(context)}')
    # print(f'[langgraph_agent_jp][llm_answer()] context: {context}')

    # last_question = "정보보호위원회 위원에 대해 알려줘."
    
    response = pdf_chain.invoke(
        {
            "question": latest_question,
            "context": context,
            # "chat_history": messages_to_history(state["messages"]),
            "chat_history": []
        }
    )

    search_results = state['context']
    print(f"[langgraph_agent_jb][llm_answer()] search_results: {search_results}")

    # if not search_results:
    #     logger.warning("No relevant documents found in search results.")
            
    linked_docs = []
    base_url = "https://jabis.jbbank.co.kr/jabis_pdf_view"
    
    for search_result in search_results:
        # if search_result[1] < 0.8:  # relevance threshold
        params = {
            "source": search_result.metadata["source"],
            "title": search_result.metadata["title"],
            "page": search_result.metadata["page"] + 1,
        }
        url_with_params = base_url + "?" + urlencode(params)
        
        linked_docs.append(
            f"👉 [{params['title']}]({url_with_params}) [pages]: {params['page']}"
        )

    # print(f"[langgraph_agent_jb][llm_answer()] linked_docs: {linked_docs}")
       
    response = response + "\n\n 📖 관련 문서 보기\n\n" + "\n\n".join(linked_docs)

    print(f"response: {response}")
    # 생성된 답변, (유저의 질문, 답변) 메시지를 상태에 저장합니다.
    return {
        "answer": response,
        "messages": [("assistant", response)]
    }

    # return {
    #     "answer": response,
    #     "messages": [("user", latest_question), ("assistant", response)]
    # }

    # return {"messages": [response]}        
    # return {"messages": [llm.invoke(state["messages"])]}

# 그래프 정의
graph_builder = StateGraph(State)

# 노드 정의
graph_builder.add_node("retrieve", retrieve_document)
graph_builder.add_node("llm_answer", llm_answer)

# 엣지 정의
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "llm_answer")
graph_builder.add_edge("llm_answer", END)

# 체크포인터 설정
memory = MemorySaver()

# 컴파일
graph = graph_builder.compile()