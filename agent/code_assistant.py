import json
import os
from google import generativeai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
generativeai.configure(api_key=api_key)

# Define tools
def create_file(filename, content=""):
    with open(filename, "w") as f:
        f.write(content)
    return {"status": "success", "message": f"File {filename} created successfully."}

def run_command(command):
    try:
        result = os.popen(command).read()
        return {"status": "success", "message": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def install_package(package):
    try:
        os.system(f"pip install {package}")
        return {"status": "success", "message": f"Package {package} installed successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def read_file(filename):
    try:
        with open(filename, "r") as f:
            return {"status": "success", "message": f.read()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Available tools dictionary
available_tools = {
    "create_file": {
        "fn": create_file,
        "description": "Creates a new file. Parameters: filename, content (optional)"
    },
    "run_command": {
        "fn": run_command,
        "description": "Executes a shell command. Parameter: command string"
    },
    "install_package": {
        "fn": install_package,
        "description": "Installs a Python package using pip. Parameter: package name"
    },
    "read_file": {
        "fn": read_file,
        "description": "Reads and returns file contents. Parameter: filename"
    }
}

# Configure Generative AI Model (Gemini)
model = generativeai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction="""You're an expert coding assistant who helps with programming tasks.
    You work in start, plan, action, observe mode.
    For user queries, plan the execution steps, select relevant tools, and provide helpful responses.

    Rules:
    1. Follow strict JSON output format
    2. Perform one step at a time
    3. Carefully analyze user requests
    4. NEVER execute sudo commands or suggest their use - they are strictly forbidden
    5. If a user requests a sudo command, immediately respond with a security warning
    6. Prioritize safety - no dangerous commands
    7. Provide clear explanations

    Output JSON format:
    {
        "step": "string (plan|action|output|observe|error)",
        "content": "string",
        "function": "tool name for actions",
        "input": "parameters for the tool"
    }

    Available tools:
    - create_file(filename, content=""): Creates a new file with optional content
    - run_command(command): Executes a shell command (sudo commands not allowed)
    - install_package(package): Installs a Python package using pip
    - read_file(filename): Reads and returns file contents. Parameter: filename

    You can also use npm commands for frontend tasks. Available examples include:
    - `npm install` to install frontend dependencies.
    - `npm run build` to compile frontend code.
    """
)

def extract_json_from_text(text):
    """Extract JSON objects from text that may contain markdown code blocks"""
    if '```json' in text:
        json_blocks = []
        parts = text.split('```json\n')
        for part in parts[1:]:  # Skip the first split as it's before any json block
            if '\n```' in part:
                json_str = part.split('\n```')[0]
                try:
                    json_blocks.append(json.loads(json_str))
                except json.JSONDecodeError:
                    continue
        return json_blocks
    return []

def is_sudo_command(command: str) -> bool:
    """Check if a command contains sudo"""
    return command.strip().lower().startswith('sudo')

def execute_action(action):
    """Execute a single action based on the parsed JSON"""
    if action.get('step') == 'plan':
        print(f"🧠 Plan: {action['content']}")
        return True
    elif action.get('step') == 'action':
        function_name = action.get('function')
        if function_name == 'run_command':
            command = action.get('input')
            if isinstance(command, dict):
                command = command.get('command', '')
            if is_sudo_command(command):
                print("🚫 Security Warning: Sudo commands are not allowed for security reasons")
                return False
        
        if function_name in available_tools:
            input_params = action.get('input', {})
            if isinstance(input_params, dict):
                result = available_tools[function_name]['fn'](**input_params)
            else:
                result = available_tools[function_name]['fn'](input_params)
            print(f"✅ Action completed: {result['message']}")
            return True
    return False

# Main function to interact with the user
def main():
    print("🤖 Coding Assistant Ready! (Type 'exit' to quit)")
    print("Available commands:")
    print("- Create/read files")
    print("- Run shell commands (sudo not allowed)")
    print("- Install Python packages")
    print("- Get coding help")

    while True:
        user_input = input("> ")

        if user_input.lower() == 'exit':
            break

        # Early check for sudo commands
        if 'sudo' in user_input.lower():
            print("🚫 Security Warning: Sudo commands are not allowed for security reasons")
            continue

        try:
            response = model.generate_content(user_input)
            text = response.candidates[0].content.parts[0].text
            
            # Extract and execute all JSON blocks in sequence
            actions = extract_json_from_text(text)
            if actions:
                for action in actions:
                    execute_action(action)
            else:
                print("❌ No valid actions found in response")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("🔎 Raw response:", response)

# Run the assistant
if __name__ == "__main__":
    main()
