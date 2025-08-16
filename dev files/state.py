from langgraph.graph import MessagesState

class AgentInputState(MessagesState):
    """
    InputState is only messages
    """

class AgentState(MessagesState):
    """Main agent state containing messages and other data"""
    intent : str
    query : str
    context : str | None
    response : str
