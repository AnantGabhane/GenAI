from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import generativeai
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os
from pathlib import Path

# Load environment variables and configure
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
generativeai.configure(api_key=api_key)

# Initialize the model
model = generativeai.GenerativeModel("gemini-1.5-pro")


def get_relevant_context(question: str):
    relevent_chunks = retriver.similarity_search(
        query=question, k=8, score_threshold=0.5
    )

    formatted_chunks = []
    for chunk in relevent_chunks:
        page_num = chunk.metadata.get("page_label", "Unknown page")
        content = chunk.page_content.strip()
        formatted_chunks.append(f"[Page {page_num}]: {content}")

    return "\n\n---\n\n".join(formatted_chunks)


def get_response(user_query: str, context_text: str) -> str:
    system_prompt = """
    You are a helpful assistant that answers questions based on the provided context from the Node.js Application Developer's Guide.
    
    Context from the document:
    {context_text}
    
    Instructions:
    1. Use only the information provided in the context above to answer questions
    2. If the information isn't in the context, say "I cannot provide complete information about [topic]. Here's what I found in the documentation: [partial information if available]"
    3. Include specific page references [Page X] when providing information
    4. Keep answers concise and focused
    5. If multiple pages contain relevant information, cite all relevant pages
    6. If the information is only partially available, specify what aspects are covered and what's missing
    """.format(
        context_text=context_text
    )

    full_prompt = f"""
{system_prompt}

User Question: {user_query}

Remember to:
- Only use information from the provided context
- Include page references
- Be specific about what information is or isn't available
"""
    response = model.generate_content(full_prompt)
    return response.text


def chat_loop():
    print("Welcome to Node.js Documentation Assistant!")
    print("Ask questions about Node.js (type 'quit' to exit)")

    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() in ["quit", "exit", "q"]:
            break

        # Get fresh context for each question
        context_text = get_relevant_context(user_question)

        # Get and print response
        response = get_response(user_question, context_text)
        print("\nResponse:", response)


if __name__ == "__main__":
    # Load and split PDF
    pdf_path = Path(__file__).parent / "node-dev.pdf"
    loader = PyPDFLoader(file_path=str(pdf_path))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(docs)

    # Initialize Google's embedding model
    embedder = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )

    # Initialize Qdrant vector store
    retriver = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="learning_langchain",
        embedding=embedder,
    )

    print("Injection Done!")
    chat_loop()


"""
(myenv) apexaiq@Anant:~/Desktop/GenAI/RAG$ python3 rag_1.py 
Injection Done!
Welcome to Node.js Documentation Assistant!
Ask questions about Node.js (type 'quit' to exit)


Your question: what is nodejs?

Response: Node.js is a software requirement for using the Node.js Client API.  You need version 6.3.1 or later.  It's available from http://nodejs.org.  The examples in the guide assume you have the `node` command on your path. [Page 14]  Node.js provides concurrency through multiple waits for IO responses, instead of multiple threads. This strategy avoids the challenges and risks of multi-threaded programming for middle-tier clients that are IO rather than compute intensive. [Page 272]  Almost all Node.js Client API operations take place through a database client object. [Page 17] You should also have the node and npm commands on your path. If you are working on Microsoft Windows, you should use a DOS command shell rather than a Cygwin shell because Cygwin is not a supported environment for node and npm. [Page 9]


Your question: why do we use nodejs?

Response: Node.js provides concurrency through multiple waits for I/O responses instead of multiple threads, avoiding the challenges of multi-threaded programming for I/O-intensive middle-tier clients [Page 272]. This allows for better throughput for large datasets by enabling multiple concurrent requests to each e-node [Page 272].  Node.js also allows developers to call methods provided by generated modules to execute endpoints on the e-node [Page 267].  The provided text also mentions using Node.js for data movement with REST API endpoints or Data Service endpoints [Page 272] and for managing user-defined code stored in the modules database [Page 258].  More capabilities are hinted at, but not detailed [Page 9].

"""
