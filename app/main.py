import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from fastapi import FastAPI
from langgraph_app.agent_graph import *
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query : str
    session_id : str | None 

class QueryResponse(BaseModel):
    response : str


@app.post("/chat")
async def call_llm(request : QueryRequest):
    messages = {"messages" : [{"role" : "user", "content" : f"{request.query}"}]}
    config = {"configurable" : {"thread_id" : request.session_id}}
    response = graph.invoke(messages,config)
    res = QueryResponse(response=response["messages"][-1].content)
    return res