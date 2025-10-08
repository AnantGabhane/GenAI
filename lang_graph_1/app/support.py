from graph import graph, create_chat_graph
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv
import json
from langgraph.types import Command

load_dotenv()

DB_URI = "mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "7"}}


def init():
    """Initialize the graph."""
    print("LangGraph Chat Initialized. Type your message or Ctrl+C to exit.")

    with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)
        state = graph_with_mongo.get_state(config=config)
        # print("state", state.values['messages'])
        # for message in state.values["messages"]:
        #     message.pretty_print()
        messages = state.values["messages"]
        user_query = None

        # Traverse messages from most recent to oldest to find the latest human_assistance_tool call
        for message in reversed(messages):
            tool_calls = message.additional_kwargs.get("tool_calls", [])
            # Fallback for providers that use single function_call field
            if not tool_calls and "function_call" in message.additional_kwargs:
                tool_calls = [{"function": message.additional_kwargs["function_call"]}]

            for call in tool_calls:
                fn = call.get("function") or {}
                if fn.get("name") == "human_assistance_tool":
                    args = fn.get("arguments")
                    # Normalize args into dict
                    if isinstance(args, dict):
                        args_dict = args
                    elif isinstance(args, str):
                        try:
                            args_dict = json.loads(args)
                        except json.JSONDecodeError:
                            # Attempt to parse "key: value" pattern as a last resort
                            args_dict = {}
                            try:
                                import re
                                m = re.search(r"query\s*:\s*(.*)", args)
                                if m:
                                    args_dict["query"] = m.group(1).strip()
                            except Exception:
                                pass
                    else:
                        args_dict = {}

                    user_query = (
                        args_dict.get("query")
                        or args_dict.get("input")
                        or args_dict.get("text")
                    )
                    if user_query:
                        break
            if user_query:
                break

        if user_query:
            print("User is trying to ask query:", user_query)
        else:
            print("User is trying to ask query: None")
        human_response = input("Resolution > ")
        resume_command = Command(resume={"data": human_response})
        for event in graph_with_mongo.stream(resume_command, config, stream_mode="values"):
            if "messages" in event:
                event["messages"][-1].pretty_print()


if __name__ == "__main__":
    init()

"""
LangGraph Chat Initialized. Type your message or Ctrl+C to exit.
================================ Human Message =================================

Remember that my name is anant gabhane
================================== Ai Message ==================================

Okay, Anant Gabhane. I will remember that. How can I help you today?
================================ Human Message =================================

what is my name ?
================================== Ai Message ==================================

Your name is Anant Gabhane.
================================ Human Message =================================

what is my name?
================================== Ai Message ==================================

Your name is Anant Gabhane.
================================ Human Message =================================

hey i'm facing problem with login , can you connect with me someone?
================================== Ai Message ==================================
Tool Calls:
  human_assistance_tool (64785d70-688e-41c1-a364-d69a91aeac7c)
 Call ID: 64785d70-688e-41c1-a364-d69a91aeac7c
  Args:
    query: User Anant Gabhane is facing problem with login, connect with him.
(app) apexaiq@Anant:~/Desktop/GenAI/lang_graph_1/app$ 
"""
