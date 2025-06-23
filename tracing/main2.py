from dotenv import load_dotenv
from google import generativeai as genai
import os
import json
import requests
import time
from datetime import datetime
from langfuse import Langfuse
from langfuse import observe, get_client

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")


# Configure APIs
genai.configure(api_key=GOOGLE_API_KEY)
os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY

# Initialize LangFuse client
try:
    client = Client()
    langfuse = get_client()
    print("Successfully connected to LangFuse")
except Exception as e:
    print(f"Error connecting to LangFuse: {e}")
    client = None

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.0-pro")  # Changed to regular model


@observe
def get_weather(city: str):
    """Function to get the weather data"""
    print("🔧 : Tool called : get_weather", city)
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    return "❌ : Something went wrong"

@observe
def generate_response_with_retry(prompt, max_retries=3, delay=60):
    """Helper function to generate response from Gemini with retry logic"""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text if response else None
        except Exception as e:
            if "429" in str(e):  # Rate limit error
                wait_time = delay * (attempt + 1)
                print(
                    f"\nRate limit reached. Waiting {wait_time} seconds before retry..."
                )
                time.sleep(wait_time)
            else:
                print(f"Error generating response: {e}")
                return None
    return None



@observe
def process_weather_query(query):
    """Process weather-related queries with tracing"""
    try:
        # Extract city name
        city = query.lower().split("weather of ")[-1].strip("?. ")

        if city:
            # Get weather information
            weather_info = get_weather(city)
            return weather_info
    except Exception as e:
        return f"Error processing weather query: {str(e)}"



@observe
def chat_loop():
    print("\nChat initialized. Type 'exit' to quit.")

    while True:
        try:
            user_query = input("\nYou: ")
            if user_query.lower() == "exit":
                break

            # For weather queries
            if "weather" in user_query.lower():
                weather_info = process_weather_query(user_query)
                print(f"🤖 : {weather_info}")
                continue

            # For other queries
            prompt = f"User Query: {user_query}\nProvide a helpful response:"
            response = generate_response_with_retry(prompt)

            if response:
                print(f"🤖 : {response}")
            else:
                print(f"❌ : Sorry, I couldn't generate a response at this time.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("Weather Agent with LangFuse Tracing")
    chat_loop()
