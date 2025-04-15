from dotenv import load_dotenv
from google import generativeai as genai
import os
import json
import subprocess
import sys

def safe_json_loads(text):
    """Parse JSON from text, handling various formats"""
    print(f"Attempting to parse: {text}")
    try:
        # First try direct JSON parsing
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # Clean up the text by removing markdown code block markers
            cleaned_text = text.replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            try:
                # Try to find and parse the last JSON object in the text
                start = text.rfind("{")
                end = text.rfind("}") + 1
                if start != -1 and end != 0:
                    json_str = text[start:end]
                    return json.loads(json_str)
            except:
                print(f"Debug - Failed to parse response: {text}")
                return {"step": "error", "content": "Failed to parse response"}

def create_file(filename: str, content: str = ""):
    """Create a new file with optional content"""
    print(f"🔧 Tool called: create_file {filename}")
    try:
        # Get absolute path
        abs_path = os.path.abspath(filename)
        print(f"Creating file at: {abs_path}")
        
        # Ensure the directory exists
        directory = os.path.dirname(abs_path)
        if directory and not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)
        
        # Write file with explicit encoding
        with open(abs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Verify file was created
        if os.path.exists(abs_path):
            print(f"✅ File created successfully at: {abs_path}")
            return f"✅ File {filename} created successfully"
        else:
            error_msg = f"❌ File creation failed: File does not exist after writing"
            print(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"❌ Error creating file: {str(e)}"
        print(f"Detailed error: {str(e)}")
        return error_msg

def run_command(command: str):
    """Execute a shell command"""
    print(f"🔧 Tool called: run_command {command}")
    
    # Check for sudo commands
    if 'sudo' in command.lower().split():
        return "❌ Security Error: Sudo commands are not allowed for safety reasons"
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return f"✅ Command executed successfully\nOutput: {result.stdout}"
        return f"❌ Command failed: {result.stderr}"
    except Exception as e:
        return f"❌ Error executing command: {str(e)}"

def install_package(package: str):
    """Install a Python package using pip"""
    print(f"🔧 Tool called: install_package {package}")
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', package],
                              capture_output=True, text=True)
        if result.returncode == 0:
            return f"✅ Package {package} installed successfully"
        return f"❌ Installation failed: {result.stderr}"
    except Exception as e:
        return f"❌ Error installing package: {str(e)}"

def read_file(filename: str):
    """Read content of a file"""
    print(f"🔧 Tool called: read_file {filename}")
    try:
        with open(filename, 'r') as f:
            content = f.read()
        return f"📄 File contents:\n{content}"
    except Exception as e:
        return f"❌ Error reading file: {str(e)}"

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

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    "gemini-1.5-flash",  # Changed from gemini-1.5-pro to gemini-1.5-flash
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
        "step": "string (plan|action|output)",
        "content": "string",
        "function": "tool name for actions",
        "input": "parameters for the tool"
    }

    Available tools:
    - create_file(filename, content=""): Creates a new file with optional content
    - run_command(command): Executes a shell command (sudo commands not allowed)
    - install_package(package): Installs a Python package using pip
    - read_file(filename): Reads and returns file contents
    """
)

def main():
    print("🤖 Coding Assistant Ready! (Type 'exit' to quit)")
    print("Available commands:")
    print("- Create/read files")
    print("- Run shell commands (sudo not allowed)")
    print("- Install Python packages")
    print("- Get coding help\n")

    while True:
        try:
            user_query = input("> ")
            if user_query.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break

            if 'sudo' in user_query.lower().split():
                print("❌ Security Error: Sudo commands are not allowed for safety reasons")
                continue

            contents = [{"role": "user", "parts": [{"text": user_query}]}]

            while True:
                try:
                    response = model.generate_content(contents=contents)
                    if not response.text:
                        print("Empty response received")
                        break

                    parsed_output = safe_json_loads(response.text)
                    print(f"Debug - Parsed output: {json.dumps(parsed_output, indent=2)}")

                    if parsed_output["step"] == "error":
                        print("Error in parsed output")
                        break

                    if parsed_output["step"] == "plan":
                        print(f"🧠 Planning: {parsed_output['content']}")
                        contents.append({"role": "model", "parts": [{"text": json.dumps(parsed_output)}]})
                        continue

                    if parsed_output["step"] == "action":
                        tool_name = parsed_output.get("function")
                        tool_input = parsed_output.get("input")

                        if tool_name in available_tools:
                            try:
                                if isinstance(tool_input, dict):
                                    result = available_tools[tool_name]["fn"](**tool_input)
                                else:
                                    result = available_tools[tool_name]["fn"](tool_input)
                                
                                print(f"Tool execution result: {result}")
                                
                                if "❌" in result:  # If there was an error
                                    break
                                    
                                contents.append({"role": "user", "parts": [{"text": json.dumps({"step": "observe", "content": result})}]})
                                continue
                            except Exception as e:
                                print(f"❌ Tool execution error: {str(e)}")
                                break

                    if parsed_output["step"] == "output":
                        print(f"✅ {parsed_output['content']}")
                        break

                except Exception as e:
                    print(f"❌ Response processing error: {str(e)}")
                    break

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Critical error: {str(e)}")

if __name__ == "__main__":
    main()
