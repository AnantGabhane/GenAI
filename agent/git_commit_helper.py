import os
import subprocess
from typing import List, Dict, Tuple
from google import generativeai as genai
from dotenv import load_dotenv

def get_unstaged_files() -> List[str]:
    """Get list of unstaged files from git"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, check=True)
        files = []
        for line in result.stdout.splitlines():
            if line.startswith('??') or line.startswith(' M') or line.startswith('M '):
                files.append(line[3:])
        return files
    except subprocess.CalledProcessError:
        print("❌ Error: Not a git repository or git command failed")
        return []

def stage_file(file_path: str) -> bool:
    """Stage a file using git add"""
    try:
        subprocess.run(['git', 'add', file_path], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error staging {file_path}: {e.stderr.decode()}")
        return False

def read_file_content(file_path: str) -> str:
    """Read content of a file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

def generate_commit_message(model, file_path: str, content: str) -> str:
    """Generate commit message using Gemini"""
    prompt = f"""
    Generate a concise commit message for the following file changes.
    File: {file_path}
    
    Rules:
    1. Start with an appropriate emoji
    2. Use present tense
    3. Be specific but concise
    4. Follow conventional commits format (type(scope): description)
    5. Max 50 characters for the first line
    
    File content:
    {content[:1000]}  # Limiting content to first 1000 chars
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error generating commit message: {str(e)}"

def commit_changes(message: str) -> Tuple[bool, str]:
    """Commit staged changes"""
    try:
        result = subprocess.run(
            ['git', 'commit', '-m', message],
            check=True,
            capture_output=True,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    # Load environment variables and configure API
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in .env file")
        return
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    
    # Get unstaged files
    files = get_unstaged_files()
    if not files:
        print("✨ No unstaged files found")
        return
    
    print("📝 Generating commit messages for unstaged files...\n")
    
    staged_files = []
    commit_messages = {}
    
    # Generate messages and stage files
    for file_path in files:
        content = read_file_content(file_path)
        commit_msg = generate_commit_message(model, file_path, content)
        
        print(f"\n🔍 File: {file_path}")
        print(f"💡 Suggested commit: {commit_msg}")
        
        # Ask user if they want to stage this file
        while True:
            choice = input(f"\nDo you want to stage {file_path}? (y/n/q to quit): ").lower()
            if choice in ['y', 'n', 'q']:
                break
            print("Please enter 'y' for yes, 'n' for no, or 'q' to quit")
        
        if choice == 'q':
            print("\n👋 Exiting...")
            return
        
        if choice == 'y':
            if stage_file(file_path):
                print(f"✅ Staged: {file_path}")
                staged_files.append(file_path)
                commit_messages[file_path] = commit_msg
            else:
                print(f"❌ Failed to stage: {file_path}")
    
    if not staged_files:
        print("\n❌ No files were staged. Exiting...")
        return
    
    # Handle commits
    print("\n📦 Staged files and their commit messages:")
    for file_path in staged_files:
        print(f"\n📄 {file_path}")
        print(f"💬 {commit_messages[file_path]}")
    
    # Ask if user wants to commit all staged files
    while True:
        commit_choice = input("\nDo you want to commit all staged files? (y/n): ").lower()
        if commit_choice in ['y', 'n']:
            break
        print("Please enter 'y' for yes or 'n' for no")
    
    if commit_choice == 'y':
        # If there's only one file, use its commit message
        # If multiple files, ask user to provide a summary commit message
        if len(staged_files) == 1:
            commit_msg = commit_messages[staged_files[0]]
        else:
            print("\n📝 Multiple files staged. Please choose a commit approach:")
            print("1. Use separate commits for each file")
            print("2. Create a single commit for all files")
            
            while True:
                approach = input("\nEnter your choice (1 or 2): ")
                if approach in ['1', '2']:
                    break
                print("Please enter '1' or '2'")
            
            if approach == '1':
                # Commit files individually
                for file_path in staged_files:
                    success, output = commit_changes(commit_messages[file_path])
                    if success:
                        print(f"\n✅ Committed {file_path}")
                        print(output)
                    else:
                        print(f"\n❌ Failed to commit {file_path}")
                        print(output)
                return
            else:
                # Get a summary commit message for all files
                print("\n💡 Generating a summary commit message...")
                files_content = "\n".join([f"{f}: {commit_messages[f]}" for f in staged_files])
                commit_msg = generate_commit_message(model, "multiple files", files_content)
        
        # Perform the commit
        success, output = commit_changes(commit_msg)
        if success:
            print("\n✅ Successfully committed changes!")
            print(output)
        else:
            print("\n❌ Failed to commit changes")
            print(output)
    else:
        print("\n💾 Changes are staged but not committed. You can commit them later.")

if __name__ == "__main__":
    main()
