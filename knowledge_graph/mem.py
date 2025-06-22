import google.generativeai as genai
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
from datetime import datetime
import os

# Configure API keys
GOOGLE_API_KEY = 'AIzaSyCd0cNIcNX82y9Mefdf5S2w4_6wQmqRgz8'
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Database configurations
QDRANT_HOST = "localhost"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "qBjk6TD4lJtOtyTLse_sL4bqnuEtjme7IiqU7CDVbjU"

# Initialize Neo4j connection
neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

# Initialize Gemini components
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.7,
    streaming=True  # Enable streaming responses
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Qdrant client and vector store
qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
try:
    # Try to create a new collection
    qdrant_client.create_collection(
        collection_name="chat_history",
        vectors_config={
            "size": 768,  # Dimension of Gemini embeddings
            "distance": "Cosine"
        }
    )
except Exception as e:
    print(f"Collection might already exist: {e}")

vector_store = Qdrant(
    client=qdrant_client,
    collection_name="chat_history",
    embeddings=embeddings
)

# Initialize conversation memory with a window of 10 messages
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    k=10  # Keep last 10 exchanges
)

# Initialize conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

def store_in_neo4j(user_message, bot_response, user_id):
    with neo4j_driver.session() as session:
        # Create nodes and relationships
        session.run("""
            MERGE (u:User {id: $user_id})
            CREATE (m:Message {content: $user_message, timestamp: datetime()})
            CREATE (r:Response {content: $bot_response, timestamp: datetime()})
            CREATE (u)-[:SENT]->(m)
            CREATE (m)-[:RECEIVED]->(r)
        """, user_id=user_id, user_message=user_message, bot_response=bot_response)

def chat(message, user_id="default_user"):
    try:
        # Search for relevant past conversations using MMR
        search_results = vector_store.max_marginal_relevance_search(
            message,
            k=3,
            fetch_k=10  # Fetch more candidates for better diversity
        )
        
        # Prepare context from search results
        context = ""
        if search_results:
            context = "Relevant context from previous conversations:\n"
            for doc in search_results:
                context += f"- {doc.page_content}\n"
            context += "\n"

        # Prepare the full prompt
        full_prompt = f"{context}\nHuman: {message}"
        
        # Get response from conversation chain
        response = conversation.predict(input=full_prompt)
        
        # Store the conversation in vector store
        vector_store.add_texts(
            texts=[f"User: {message}\nAssistant: {response}"],
            metadatas=[{"user_id": user_id, "timestamp": str(datetime.now())}]
        )
        
        # Store in Neo4j
        store_in_neo4j(message, response, user_id)
        
        return response

    except Exception as e:
        print(f"Error: {str(e)}")
        return "I apologize, but I encountered an error. Please try again."

def main():
    print("Enhanced Chat with Gemini (type 'exit' to quit)")
    print("Features:")
    print("- Memory-enabled conversations")
    print("- Context-aware responses")
    print("- Graph-based relationship tracking")
    print("- Vector similarity search")
    print("\nInitializing...")
    
    user_id = input("Please enter your user ID (or press Enter for default): ").strip() or "default_user"
    
    while True:
        try:
            message = input("\nYou: ")
            if message.lower() == 'exit':
                print("Goodbye!")
                break
            
            response = chat(message=message, user_id=user_id)
            print(f"Bot: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    from datetime import datetime
    main()
    # Clean up connections
    neo4j_driver.close()