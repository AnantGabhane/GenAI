from dotenv import load_dotenv
from google import generativeai as genai  # This should work now
import os
import argparse


def setup_argparse():
    parser = argparse.ArgumentParser(description="Chat with Gemini AI")
    parser.add_argument("--prompt", "-p", type=str, help="The prompt to send to the AI")
    parser.add_argument(
        "--system",
        "-s",
        type=str,
        default="You are a helpful assistant that provides clear and concise answers.",
        help="System instruction for the AI",
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


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    # Load environment variables and configure API
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

    # Create model instance
    model = genai.GenerativeModel(args.model, system_instruction=args.system)

    # Start chat
    chat = model.start_chat(history=[])

    if args.interactive:
        print("Starting interactive chat (type 'quit' to exit)")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            try:
                response = chat.send_message(user_input)
                print("\nAI:", response.text)
            except Exception as e:
                print(f"Error: {e}")
    else:
        if not args.prompt:
            parser.error("--prompt is required when not in interactive mode")
        try:
            response = chat.send_message(args.prompt)
            print(response.text)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()


"""
--prompt "what is ai"
AI, or Artificial Intelligence, refers to the simulation of human intelligence processes by machines, especially computer systems.  These processes include learning (acquiring information and rules for using the information), reasoning (using rules to reach approximate or definite conclusions), and self-correction.
"""
