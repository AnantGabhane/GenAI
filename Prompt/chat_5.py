from dotenv import load_dotenv
from google import generativeai as genai
import os
from collections import Counter
import argparse
import json


def setup_argparse():
    parser = argparse.ArgumentParser(description="Self-Consistency Chat with Gemini AI")
    parser.add_argument("--prompt", "-p", type=str, help="The prompt to send to the AI")
    parser.add_argument(
        "--num-generations",
        "-n",
        type=int,
        default=5,
        help="Number of responses to generate for consistency check",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemini-1.5-flash",
        choices=["gemini-1.5-flash", "gemini-1.5-pro"],
        help="Model to use",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Enable interactive chat mode"
    )
    return parser


def generate_multiple_responses(model, prompt, num_generations):
    responses = []
    for _ in range(num_generations):
        chat = model.start_chat(history=[])
        response = chat.send_message(prompt)
        # Extract the result content from the response
        try:
            response_dict = json.loads(
                response.text.split("result", 1)[1].split("}", 1)[0] + "}"
            )
            result = response_dict.get("content", response.text)
            responses.append(result)
        except:
            responses.append(response.text)
    return responses


def find_most_consistent_response(responses):
    # Use Counter to find the most common response
    response_counts = Counter(responses)
    most_common = response_counts.most_common(1)[0]

    return {
        "most_common_response": most_common[0],
        "frequency": most_common[1],
        "total_responses": len(responses),
        "consistency_score": most_common[1] / len(responses),
        "all_responses": responses,
    }


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    # Load environment variables and configure API
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

    # Create model instance with system instruction for structured output
    system_instruction = """You're a helpful assistant who provides clear and consistent answers.
    Always follow this structured thinking process:
    1. Analyze the question carefully
    2. Think about different aspects of the answer
    3. Provide a clear, concise response
    4. Format the output as: {{step: "result", content: "your final answer"}}
    """

    model = genai.GenerativeModel(args.model, system_instruction=system_instruction)

    if args.interactive:
        print("Starting interactive chat (type 'quit' to exit)")
        print(
            f"Generating {args.num_generations} responses for each prompt to ensure consistency"
        )
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            try:
                responses = generate_multiple_responses(
                    model, user_input, args.num_generations
                )
                result = find_most_consistent_response(responses)

                print("\nAI: ", result["most_common_response"])
                print(f"\nConsistency Score: {result['consistency_score']:.2f}")
                print(
                    f"This answer appeared {result['frequency']} times out of {result['total_responses']} generations"
                )

                if result["consistency_score"] < 0.5:
                    print(
                        "\nNote: Low consistency in responses. The answer might be uncertain or variable."
                    )

            except Exception as e:
                print(f"Error: {e}")
    else:
        if not args.prompt:
            parser.error("--prompt is required when not in interactive mode")
        try:
            responses = generate_multiple_responses(
                model, args.prompt, args.num_generations
            )
            result = find_most_consistent_response(responses)

            print("\nMost Consistent Response:", result["most_common_response"])
            print(f"Consistency Score: {result['consistency_score']:.2f}")
            print(f"Frequency: {result['frequency']}/{result['total_responses']}")

            if result["consistency_score"] < 0.5:
                print(
                    "\nNote: Low consistency in responses. The answer might be uncertain or variable."
                )

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()

"""

Most Consistent Response: {{step: "result", content: "Artificial intelligence (AI) is a broad field of computer science dedicated to creating systems capable of performing tasks that typically require human intelligence.  These tasks include learning, reasoning, problem-solving, perception, and natural language understanding."}}

Consistency Score: 0.60
Frequency: 3/5
"""
