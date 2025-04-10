from dotenv import load_dotenv
from google import generativeai as genai
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Create a model instance
model = genai.GenerativeModel("gemini-1.5-pro")

# Start a chat
chat = model.start_chat(
    history=[],
)

# Define the context and question in a single message
message = """Context: You are a helpful assistant that answers questions about Shegaon, Maharashtra."""

# Generate the response
response = chat.send_message("What is Shegaon known for?")
print(response.text)

""" context provided in system message with question in system message
Shegaon is primarily known for the **Shri Sant Gajanan Maharaj Sansthan**, a large and popular Hindu temple complex dedicated to Sant Gajanan Maharaj.  This draws a significant number of pilgrims throughout the year.  The Sansthan also runs a number of charitable institutions including hospitals, educational institutes, and anand sagar project (artificial lake).  Beyond the religious aspect, Shegaon is also known for its associated businesses catering to the pilgrims, such as lodging, restaurants, and shops selling religious items.
"""

""" context provided in system message with question in user message
Shegaon, in the Buldhana district of Maharashtra, India, is primarily known for the **Shri Gajanan Maharaj Sansthan**.  This is a large and popular Hindu temple complex dedicated to Gajanan Maharaj, a saint of the late 19th and early 20th centuries.  The Sansthan not only manages the temple but also runs a number of charitable institutions including hospitals, educational institutions, and annadan (free food distribution).

So, in short, Shegaon is famous as a significant pilgrimage center for devotees of Gajanan Maharaj.
"""
