from dotenv import load_dotenv
from google import generativeai as genai
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Create a model instance with Gemini 1.5 Flash
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction="""You're a helpful assistant who is expert in breaking down complex problems and then resolve the user query 
    for the given user input,analyse the input and break down the problem step by step.Atleast think 5-6 steps on how to solve the problem
    before solving it down.

    The steps are you get a user input, you analyse, you think and then you again think for several times and then return an output with explaination and 
    then you finally validate the output as well before returning it.

    follow these steps in sequence that is "analyse,think,think,think,think,think,output,validate,validate,result"

    Rules:
    1. Follow the strict plain text output format.
    2. Always perform one step at a time and wait for next input 
    3. carefully analyse the user query

    Output format : 
    {{step : "string", content: "string"}}

    Example : 
    Input : what is 2+2?
    Output :{{step : "analyse", content: "Alright! the user is intrested in maths query and he's asking a basic arithmatic operation"}}
    Output :{{step : "think", content: "To perform the addition i must go from left to right and add all the oprands"}} 
    Output :{{step : "output", content: "4"}} 
    Output :{{step : "validate", content: "seems like 4 is the correct answer for 2+2"}} 
    Output :{{step : "validate", content: "seems like 4 is the correct answer for 2+2"}} 
    Output :{{step : "result", content: "2 +2 =  4 and that is calculated by adding all numbers"}} 

    """,
)



# Start a chat
chat = model.start_chat(history=[])
# Send the question
# response = chat.send_message("What is love?")  # I cannot answer this question.
response = chat.send_message("What is love?")
print(response.text)

# Print response
"""
response:
GenerateContentResponse(
    done=True,
    iterator=None,
    result=protos.GenerateContentResponse({
      "candidates": [
        {
          "content": {
            "parts": [
              {
                "text": "{{step : \"analyse\", content: \"The user is asking for the result of a mathematical expression involving addition and multiplication.\"}}\n{{step : \"think\", content: \"Order of operations (PEMDAS/BODMAS) needs to be considered. Multiplication comes before addition.\"}}\n{{step : \"think\", content: \"First, perform the multiplication: 4 * 5 = 20.\"}}\n{{step : \"think\", content: \"Then, perform the addition: 3 + 20 = 23.\"}}\n{{step : \"think\", content: \"Therefore, the final answer should be 23.\"}}\n{{step : \"think\", content: \"I need to verify the calculation to ensure accuracy.\"}}\n{{step : \"output\", content: \"23\"}}\n{{step : \"validate\", content: \"Following the order of operations, the calculation is correct.\"}}\n{{step : \"validate\", content: \"The answer 23 is consistent with the rules of arithmetic.\"}}\n{{step : \"result\", content: \"3 + 4 * 5 = 23.  The multiplication 4 * 5 is performed first (resulting in 20), then the addition 3 + 20 is done, yielding the final answer 23.\"}}\n"
              }
            ],
            "role": "model"
          },
          "finish_reason": "STOP",
          "avg_logprobs": -0.09897297016088513
        }
      ],
      "usage_metadata": {
        "prompt_token_count": 396,
        "candidates_token_count": 276,
        "total_token_count": 672
      },
      "model_version": "gemini-1.5-flash"
    }),
)
"""

# Prints response.to_dict()
"""
{'candidates': [{'content': {'parts': [{'text': '{{step : "analyse", content: "The user is asking for the solution to a mathematical expression involving addition and multiplication."}}\n{{step : "think", content: "The expression 3 + 4 * 5 needs to be evaluated according to the order of operations (PEMDAS/BODMAS), which prioritizes multiplication before addition."}}\n{{step : "think", content: "First, perform the multiplication: 4 * 5 = 20"}}\n{{step : "think", content: "Next, perform the addition: 3 + 20 = 23"}}\n{{step : "think", content: "Therefore, the solution should be 23."}}\n{{step : "think", content: "I need to verify the calculation to ensure accuracy."}}\n{{step : "output", content: "23"}}\n{{step : "validate", content: "Following the order of operations, 4*5 is performed first resulting in 20. Then, 3 is added to 20 resulting in 23. The calculation is correct."}}\n{{step : "validate", content: "The answer 23 is consistent with the order of operations rules."}}\n{{step : "result", content: "3 + 4 * 5 = 23.  This is obtained by first multiplying 4 and 5 (following the order of operations), and then adding the result to 3."}}\n\n'}], 'role': 'model'}, 'finish_reason': 1, 'avg_logprobs': -0.12667880229142278, 'safety_ratings': [], 'token_count': 0, 'grounding_attributions': []}], 'usage_metadata': {'prompt_token_count': 396, 'candidates_token_count': 307, 'total_token_count': 703, 'cached_content_token_count': 0}, 'model_version': 'gemini-1.5-flash'}
"""

# JSON FORMAT OUTPUT RULE
"""
```json
{{step : "analyse", content: "The user is asking for the result of a mathematical expression involving addition and multiplication."}}
```
```json
{{step : "think", content: "To solve this, we need to follow the order of operations (PEMDAS/BODMAS), which prioritizes multiplication before addition."}}
```
```json
{{step : "think", content: "First, we perform the multiplication: 4 * 5 = 20."}}
```
```json
{{step : "think", content: "Then, we perform the addition: 3 + 20 = 23."}}
```
```json
{{step : "think", content: "Therefore, the final answer should be 23."}}
```
```json
{{step : "think", content: "Let's verify this using a calculator or another method to ensure accuracy."}}
```
```json
{{step : "output", content: "23"}}
```
```json
{{step : "validate", content: "The calculation 3 + 4 * 5 = 23 is correct according to the order of operations."}}
```
```json
{{step : "validate", content: "A calculator confirms the result as 23."}}
```
```json
{{step : "result", content: "3 + 4 * 5 = 23.  This is calculated by first multiplying 4 and 5 (following order of operations), and then adding the result to 3."}}
```

"""

# PLAIN TEXT FORMAT OUTPUT RULE
"""
{{step : "analyse", content: "The user is asking for the result of a mathematical expression involving addition and multiplication."}}
{{step : "think", content: "The expression 3 + 4 * 5 involves both addition and multiplication.  I need to remember the order of operations (PEMDAS/BODMAS)."}}
{{step : "think", content: "According to PEMDAS/BODMAS, multiplication should be performed before addition."}}
{{step : "think", content: "First, calculate 4 * 5 which equals 20."}}
{{step : "think", content: "Next, add 3 to the result of the multiplication (20)." }}
{{step : "think", content: "Finally, compute 3 + 20."}}
{{step : "output", content: "23"}}
{{step : "validate", content: "Let's check: 4 * 5 = 20, 20 + 3 = 23. The answer seems correct."}}
{{step : "validate", content: "The order of operations was correctly followed.  The calculation is accurate."}}
{{step : "result", content: "3 + 4 * 5 = 23.  Multiplication is done before addition according to the order of operations."}}
"""
