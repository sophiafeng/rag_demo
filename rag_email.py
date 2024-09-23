from dotenv import load_dotenv
from custom_gmail_reader import CustomGmailReader
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from langfuse.llama_index import LlamaIndexCallbackHandler

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

load_dotenv()

# ... rest of the code ...
 
langfuse_callback_handler = LlamaIndexCallbackHandler()
Settings.callback_manager = CallbackManager([langfuse_callback_handler])

if not langfuse_callback_handler.auth_check():
    print("Authentication failed. Please check your credentials.")
    exit(1)

# Instantiate the CustomGmailReader
loader = CustomGmailReader(
    query="",
    max_results=10,
    results_per_page=2,
    service=None
)

# Load the emails
documents = loader.load_data()

# Print email information
print(f"Number of documents: {len(documents)}")
for i, doc in enumerate(documents[:20]):
    print(f"Document {i+1}:")
    print(f"To: {doc.metadata.get('to', 'N/A')}")
    print(f"From: {doc.metadata.get('from', 'N/A')}")
    print(f"Subject: {doc.metadata.get('subject', 'N/A')}")
    print(f"Date: {doc.metadata.get('date', 'N/A')}")
    print(f"Content snippet: {doc.text[:1000]}...")
    print("=" * 50)

# Create index
index = VectorStoreIndex.from_documents(documents)

# Create retriever
retriever = VectorIndexRetriever(index=index)

# Create query engine
query_engine = RetrieverQueryEngine(retriever=retriever)

# Example query
# response = query_engine.query("Did I get off the waitlist for any of the Solidcore classes?")
response = query_engine.query("DId the latest app get accepted by the app store?")
print(response)

langfuse_callback_handler.flush()