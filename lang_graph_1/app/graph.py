"""
uv pip install langchain
uv pip install -U "langchain[google-genai]"
"""

import os
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.types import interrupt
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set. Please set it in your .env file."
    )
os.environ["GOOGLE_API_KEY"] = api_key


@tool()
def human_assistance_tool(query: str):
    """Request assistance from a human support agent for tasks that require manual intervention, such as account changes, technical issues, or access problems that cannot be resolved automatically. 
    
    Only use this tool when:
    - User explicitly requests to speak with a human/support agent
    - User reports technical issues (login problems, system errors, bugs)
    - User requests account modifications (email changes, password resets, account deletions)
    - User needs help with tasks that require manual verification or approval
    
    Do NOT use this tool for:
    - General questions that can be answered from conversation history
    - Status updates on previously completed requests
    - Simple informational queries
    """
    human_response = interrupt(
        {"query": query}
    )  # Graph will exit out after saving data in database
    return human_response["data"]  # resume with the data


tools = [human_assistance_tool]
# Register the actual tool instances (not the decorator symbol)
tool_node = ToolNode(tools=tools)
# Initialize the chat model
llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools=tools)


class State(TypedDict):
    """State of the graph.
    add_messages > adds messages back to the graph
    """

    messages: Annotated[list, add_messages]


def chatbot(state: State):
    user_content = state["messages"][-1].content.lower()
    if "status" in user_content or "what's" in user_content:
        # Find the last tool response
        for i in range(len(state["messages"]) - 2, -1, -1):
            msg = state["messages"][i]
            if msg.type == "tool":
                # Return the tool response as AI message
                return {"messages": [AIMessage(content=msg.content)]}
        # If no tool response found
        return {"messages": [AIMessage(content="No previous request status available.")]}
    else:
        message = llm_with_tools.invoke(state["messages"])
        assert len(message.tool_calls) <= 1
        return {"messages": [message]}


# build graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

# graph without any memory
graph = graph_builder.compile()


# creates a new graph with given checkpointer
def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
