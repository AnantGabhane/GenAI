from dotenv import load_dotenv
from google import generativeai as genai
import os
import json
import requests


def safe_json_loads(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If the response isn't valid JSON, try to extract JSON from the text
        try:
            # Find content between first { and last }
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                return json.loads(json_str)
        except:
            print(f"Debug - Raw response: {text}")
            return {"step": "error", "content": "Failed to parse response"}


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)


def get_weather(city: str):
    """function to get the weather data"""
    print("🔧 : Tool called : get_weather", city)
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    return "❌ : Something went wrong"


def add(a, b):
    print("🔧 : Tool called : add ", a, b)
    return a + b


available_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "takes city as an input and returns the current weather for a city",
    },
    "add": {
        "fn": add,
        "description": "takes 2 numbers as an input and returns the sum",
    },
}

model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction=f"""You're a helpful assistant who is expert in breaking down complex user queries and then resolving them
    You work on start, plan, action, observe mode.
    for the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevent tool from the available tools and based on the tool selection you perform an action to call the tool.
    wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    1. Follow the strict plain text output format.
    2. Always perform one step at a time and wait for next input 
    3. carefully analyse the user query

    Output JSON format : 
    {{
    "step" : "string"
    "content": "string"  
    "function": "The name of the function if the step is action"
    "input": "The input parameter for the function"
    }}

    Available tools :
    - get_weather(city): takes city as an input and returns the current weather for a city
    - add(a,b): takes 2 numbers as an input and returns the sum
    
    Example : 
    User query : what is current weather of new york ?
    output: {{"step" : "plan", "content": "Alright! the user is intrested in weather query and he's asking for current weather data of new york"}}
    output: {{"step" : "plan", "content": "From the available tools I should call get_weather tool to get the current weather data of new york"}}
    output: {{"step" : "action", "function": "get_weather",input: "new york"}}
    output: {{"step" : "observe", "output": "12 degree celcius"}}
    output: {{"step" : "output", "content": "The weather for new york seems to be 12 degree celcius"}}

 

    """,
)


contents = [
    {"role": "user", "parts": [{"text": "What is current weather of patiala?"}]},
    {
        "role": "user",
        "parts": [
            {
                "text": json.dumps(
                    {
                        "step": "plan",
                        "content": "The user is asking for the current weather in Patiala. I need a weather API or a similar tool to fulfill this request.  I'll need to determine if I have access to such a tool.",
                    }
                )
            }
        ],
    },
    {
        "role": "user",
        "parts": [
            {
                "text": json.dumps(
                    {
                        "step": "plan",
                        "content": "I have the `get_weather(city)` function available. I will use this function to get the current weather for Patiala.",
                    }
                )
            }
        ],
    },
    {
        "role": "user",
        "parts": [
            {
                "text": json.dumps(
                    {
                        "step": "action",
                        "function": "get_weather",
                        "input": "Patiala",
                    }
                )
            }
        ],
    },
    {
        "role": "user",
        "parts": [
            {"text": json.dumps({"step": "observe", "output": "25 degrees Celsius"})}
        ],
    },
]
while True:
    user_query = input("> ")
    contents = [{"role": "user", "parts": [{"text": user_query}]}]

    while True:
        try:
            response = model.generate_content(contents=contents)
            if not response.text:
                print("Empty response received")
                break

            parsed_output = safe_json_loads(response.text)
            if parsed_output.get("step") == "error":
                break

            contents.append(
                {"role": "user", "parts": [{"text": json.dumps(parsed_output)}]}
            )

            if parsed_output["step"] == "plan":
                print(f"🧠 : {parsed_output['content']}")
                continue

            if parsed_output["step"] == "action":
                tool_name = parsed_output.get("function")
                tool_input = parsed_output.get("input")

                if tool_name in available_tools:
                    output = available_tools[tool_name]["fn"](tool_input)
                    contents.append(
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {"step": "observe", "output": output}
                                    )
                                }
                            ],
                        }
                    )
                    continue

            if parsed_output["step"] == "output":
                print(f"🤖 : {parsed_output['content']}")
                break

        except Exception as e:
            print(f"Error: {str(e)}")
            break

"""
>  What is the weather in patiala?
🧠 : The user wants to know the current weather in Patiala.  This requires accessing weather information.
🧠 : I will use the `get_weather` tool to retrieve the current weather for Patiala.
🤖 : The current weather in Patiala is 12 degrees Celsius.



>  What is the weather in delhi?      
🧠 : The user is asking for the current weather in Delhi.  This requires accessing a weather data source.
🧠 : I will use the `get_weather` tool to retrieve the current weather for Delhi.
🔧 : Tool called : get_weather Delhi
🤖 : The current weather in Delhi is 12 degrees Celsius.


> what is the weather of mumbai in f?
🧠 : The user is asking for the current weather in Mumbai, specifically in Fahrenheit.
🧠 : To fulfill this request, I need to use the `get_weather` function with the city parameter set to "Mumbai".  Then, I will need to convert the temperature from Celsius (likely the default unit of the `get_weather` function) to Fahrenheit.
🔧 : Tool called : get_weather Mumbai
🧠 : The `get_weather` function returned a temperature of 29°C.  I need to convert this to Fahrenheit using the formula: F = (C * 9/5) + 32
🧠 : I don't have a celsius_to_fahrenheit function. I will perform the conversion directly.
🤖 : The weather in Mumbai is Haze and 84.2°F
"""
