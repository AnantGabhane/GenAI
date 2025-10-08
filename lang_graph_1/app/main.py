from graph import graph, create_chat_graph
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv

load_dotenv()

DB_URI = "mongodb://admin:admin@localhost:27017"
config = {"configurable": {"thread_id": "7"}}


def init():
    """Initialize the graph."""
    print("LangGraph Chat Initialized. Type your message or Ctrl+C to exit.")

    with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)
        try:
            while True:
                try:
                    user_input = input(">: ")
                    for event in graph_with_mongo.stream(
                        {"messages": [{"role": "user", "content": user_input}]},
                        config,
                        stream_mode="values",
                    ):
                        if "messages" in event:
                            event["messages"][-1].pretty_print()
                    # print("Bot:", result)
                except EOFError:
                    print("\nEOF detected. Exiting gracefully.")
                    break
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Exiting gracefully.")


if __name__ == "__main__":
    init()


# Output

"""
LangGraph Chat Initialized. Type your message or Ctrl+C to exit.

>: Hey remember that my name is anant 
Bot: {'messages': [HumanMessage(content='Hey remember that my name is anant ', additional_kwargs={}, response_metadata={}, id='b4b66512-c92f-4e93-a299-54d23004ea13'), AIMessage(content='Okay, Anant. I will remember that your name is Anant.', additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--7663e5f9-b2af-4430-a716-c7ea0f283c0d-0', usage_metadata={'input_tokens': 9, 'output_tokens': 16, 'total_tokens': 25, 'input_token_details': {'cache_read': 0}})]}

>: what is my name?
Bot: {'messages': [HumanMessage(content='what is my name?', additional_kwargs={}, response_metadata={}, id='36fdc80d-8bfa-4995-84f6-b546f7b23e28'), AIMessage(content="As an AI, I don't know your name. You haven't told me! I'm here to help, but I don't have access to personal information.", additional_kwargs={}, response_metadata={'prompt_feedback': {'block_reason': 0, 'safety_ratings': []}, 'finish_reason': 'STOP', 'model_name': 'gemini-2.0-flash', 'safety_ratings': []}, id='run--5a26f12e-f365-47ce-83f4-274ca80659f3-0', usage_metadata={'input_tokens': 5, 'output_tokens': 38, 'total_tokens': 43, 'input_token_details': {'cache_read': 0}})]}


After implementing preety print 

LangGraph Chat Initialized. Type your message or Ctrl+C to exit.
>: Hey remember that my name is anant 
================================ Human Message =================================

Hey remember that my name is anant 
================================== Ai Message ==================================

Okay, Anant. I will remember that your name is Anant.
>: what is my name?
================================ Human Message =================================

what is my name?
================================== Ai Message ==================================

As an AI, I don't have access to personal information, so I don't know your name. You haven't told me!
>: 
"""

# python3 main.py 
"""
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1759928496.413019 1175792 alts_credentials.cc:93] ALTS creds ignored. Not running on GCP and untrusted ALTS is not enabled.
LangGraph Chat Initialized. Type your message or Ctrl+C to exit.
>: I want to change my email from anant@g to a@g
================================ Human Message =================================

I want to change my email from anant@g to a@g
================================== Ai Message ==================================
Tool Calls:
  human_assistance_tool (c5ac8b5e-ed7d-4713-a7f3-0ee51728917e)
 Call ID: c5ac8b5e-ed7d-4713-a7f3-0ee51728917e
  Args:
    query: change my email from anant@g to a@g
>: what's the status
================================ Human Message =================================

what's the status
================================== Ai Message ==================================

we've updated the email
>: my name is anant i want to reach out to support person for logging
================================ Human Message =================================

my name is anant i want to reach out to support person for logging
================================== Ai Message ==================================
Tool Calls:
  human_assistance_tool (b9bbdc4d-fe7c-42aa-bf32-55bb01faf6e7)
 Call ID: b9bbdc4d-fe7c-42aa-bf32-55bb01faf6e7
  Args:
    query: Anant needs help with login issues.
>: what's the status
================================ Human Message =================================

what's the status
================================== Ai Message ==================================

we're currently in maintaince



"""
#$ python3 support.py 
"""
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1759928553.714367 1176960 alts_credentials.cc:93] ALTS creds ignored. Not running on GCP and untrusted ALTS is not enabled.
LangGraph Chat Initialized. Type your message or Ctrl+C to exit.
User is trying to ask query: change my email from anant@g to a@g
Resolution > we've updated the email            
================================== Ai Message ==================================
Tool Calls:
  human_assistance_tool (c5ac8b5e-ed7d-4713-a7f3-0ee51728917e)
 Call ID: c5ac8b5e-ed7d-4713-a7f3-0ee51728917e
  Args:
    query: change my email from anant@g to a@g
================================= Tool Message =================================
Name: human_assistance_tool

we've updated the email
================================== Ai Message ==================================

OK. I have updated your email from anant@g to a@g in our system.
(lang_graph_1) apexaiq@Anant:~/Desktop/GenAI/lang_graph_1/app$ python3 support.py 
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1759928610.188902 1178077 alts_credentials.cc:93] ALTS creds ignored. Not running on GCP and untrusted ALTS is not enabled.
LangGraph Chat Initialized. Type your message or Ctrl+C to exit.
User is trying to ask query: Anant needs help with login issues.
Resolution > we're currently in maintaince 
================================== Ai Message ==================================
Tool Calls:
  human_assistance_tool (b9bbdc4d-fe7c-42aa-bf32-55bb01faf6e7)
 Call ID: b9bbdc4d-fe7c-42aa-bf32-55bb01faf6e7
  Args:
    query: Anant needs help with login issues.
================================= Tool Message =================================
Name: human_assistance_tool

we're currently in maintaince 
================================== Ai Message ==================================

OK. It seems there are currently maintenance issues.
"""