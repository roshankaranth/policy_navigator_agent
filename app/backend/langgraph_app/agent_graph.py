from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import MessagesState, StateGraph, START, END
from IPython.display import Image,display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod,NodeStyles
from langchain_core.messages import HumanMessage, ToolMessage, RemoveMessage, SystemMessage
from tavily import TavilyClient
from langgraph.prebuilt import tools_condition, ToolNode
import os
from rag_pipeline import *
from langgraph.checkpoint.memory import MemorySaver
from langgraph_app.prompts import *
from langgraph_app.state import *

_ = load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

def format_chat_history(messages: list[dict]) -> str:
    return "\n".join(
        f"{m.role.capitalize()}: {m['content'].strip()}" for m in messages
    )

def web_search_tool(query : str) -> str:
    """Search on the web for answers

    Args: 
        query : str
    """
    print("Searching the web..")
    response = tavily_client.search(query)
    return response["results"]

def context_retriever(query : str) -> str:
    """ Fectches documents from vectorDB

    Args:
        query : str
    """
    retrieved_docs = retrival_pipeline(query)
    return retrieved_docs

tools = [web_search_tool, context_retriever]
llm_with_tools = llm.bind_tools(tools)

def llm_node(state : AgentState):
    clean_messages = state["messages"]
    
    if state["intent"] == "eli5":
        prompt = ELI5_Prompt.format(chat_history = clean_messages)
    elif state["intent"] == "extract_entities":
        prompt = Entity_Extraction_Prompt.format(chat_history = clean_messages)
    elif state["intent"] == "general_qa":
        prompt = general_qa_prompt.format(chat_history = clean_messages)
    elif state["intent"] == "policy_comparison":
        prompt = Policy_Comparison_Prompt.format(chat_history = clean_messages)

    response = llm_with_tools.invoke(prompt)
    return {"response" : response.content, "messages" : response}

def intent_handler(state : AgentInputState):
    user_query = state["messages"][-1].content
    prompt = Intent_Handler_Prompt.format(query = user_query)
    response = llm.invoke(prompt)
    intent = response.content
    return {"intent" : intent, "query" : user_query, "messages" : response}

def summarize_node(state : AgentInputState):
    if(len(state["messages"]) > 10):
        prompt = summary_prompt.format(last_9_messages = state["messages"][:-2])
        summary = llm.invoke(prompt)
        system_message = f"Summary of conversation earlier : {summary}"
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        delete_messages = [SystemMessage(content=system_message)] + delete_messages
        return {"messages" : delete_messages}

    return

builder = StateGraph(AgentState, input_schema = AgentInputState)
memory = MemorySaver()
tool_node = ToolNode(tools)
config = {"configurable" : {"thread_id" : "1"}}

builder.add_node("summarize_node", summarize_node)
builder.add_node("intent_handler", intent_handler)
builder.add_node("llm_node", llm_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "intent_handler")
builder.add_edge("intent_handler","summarize_node")
builder.add_edge("summarize_node", "llm_node")
builder.add_conditional_edges("llm_node", tools_condition)
builder.add_edge("tools", "llm_node")

graph = builder.compile(checkpointer=memory)
