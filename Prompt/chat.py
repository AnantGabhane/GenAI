from dotenv import load_dotenv
from google import generativeai as genai
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Create a model instance
model = genai.GenerativeModel("gemini-1.5-pro")

# Generate content
# response = model.generate_content("Explain how AI works")
response = model.generate_content("hey") # Hey there! How can I help you today?


print(response.text)

"""
AI, or Artificial Intelligence, aims to mimic human cognitive functions like learning, problem-solving, and decision-making. It achieves this through a variety of techniques, generally falling under these categories:

**1. Machine Learning (ML):**  This is the most prevalent type of AI today.  Instead of explicit programming, ML algorithms learn from data.  They identify patterns, make predictions, and improve their performance over time without human intervention. Key ML approaches include:

* **Supervised Learning:**  The algorithm learns from labeled data (input with corresponding correct output).  Think of it like training a dog with treats for good behavior. Examples include image recognition and spam filtering.
* **Unsupervised Learning:** The algorithm learns from unlabeled data, identifying patterns and structures on its own. Examples include customer segmentation and anomaly detection.
* **Reinforcement Learning:** The algorithm learns through trial and error, receiving rewards for desired actions and penalties for undesired ones.  This is how AI learns to play complex games like Go or control robots in dynamic environments.

**2. Deep Learning (DL):** A subset of ML, deep learning uses artificial neural networks with multiple layers (hence "deep") to analyze data.  These networks are inspired by the human brain and are particularly effective for complex tasks like natural language processing and image generation.  Examples include language translation and self-driving cars.

**3. Natural Language Processing (NLP):**  This branch of AI focuses on enabling computers to understand, interpret, and generate human language.  It powers applications like chatbots, voice assistants, and sentiment analysis.

**4. Computer Vision:** This field enables computers to "see" and interpret images and videos, much like humans do.  Applications include object recognition, facial recognition, and medical image analysis.

**5. Expert Systems:** These systems mimic the decision-making abilities of human experts in specific domains. They use a knowledge base of facts and rules to provide advice or solutions.  Examples include medical diagnosis systems and financial planning tools.


**How it works in practice:**

1. **Data Collection:**  AI algorithms need data to learn from. This data can be anything from text and images to sensor readings and financial transactions.
2. **Data Preprocessing:**  The collected data is cleaned, formatted, and transformed into a usable format for the AI algorithm.
3. **Model Training:**  The AI algorithm is trained on the prepared data, learning patterns and relationships within the data.
4. **Model Evaluation:**  The trained model is tested on a separate dataset to evaluate its performance and accuracy.
5. **Deployment and Monitoring:**  The model is deployed to perform its intended task and its performance is continuously monitored and refined.


**Key Concepts:**

* **Algorithms:** Sets of instructions that tell the computer what to do.
* **Neural Networks:** Interconnected nodes that process and transmit information, mimicking the human brain.
* **Training Data:** Data used to train the AI model.
* **Bias:**  Systematic errors in the AI system that can lead to unfair or inaccurate outcomes.  This can arise from biased training data or flawed algorithms.


It's important to remember that AI is not a single technology but a collection of techniques and approaches aimed at creating intelligent systems. The field is constantly evolving, with new breakthroughs and applications emerging regularly.
"""
