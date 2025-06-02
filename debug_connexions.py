"""Debug script to test connections and configurations."""
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

print("=" * 50)
print("RAG System Debug Tool")
print("=" * 50)

# Check environment variables
print("\n1. Checking environment variables...")
env_vars = {
    "WEAVIATE_URL": os.getenv("WEAVIATE_URL"),
    "WEAVIATE_API_KEY": os.getenv("WEAVIATE_API_KEY"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
}

missing = []
for var, value in env_vars.items():
    if value:
        print(f"✓ {var}: {'*' * 10}{value[-10:] if len(value) > 10 else value}")
    else:
        print(f"✗ {var}: Not set")
        missing.append(var)

if missing:
    print(f"\n⚠️  Missing environment variables: {', '.join(missing)}")
    print("Please create a .env file with the required configuration.")
    sys.exit(1)

# Test Weaviate connection
print("\n2. Testing Weaviate connection...")
try:
    import weaviate
    client = weaviate.connect_to_wcs(
        cluster_url=env_vars["WEAVIATE_URL"],
        auth_credentials=weaviate.auth.AuthApiKey(env_vars["WEAVIATE_API_KEY"]),
        headers={
            "X-OpenAI-Api-Key": env_vars["OPENAI_API_KEY"]
        }
    )
    print("✓ Successfully connected to Weaviate")
    
    # Check collections
    collections = client.collections.list_all()
    print(f"✓ Found {len(collections)} collections")
    for collection in collections:
        print(f"  - {collection}")
    
    client.close()
except Exception as e:
    print(f"✗ Weaviate connection failed: {str(e)}")
    print("\nPossible issues:")
    print("- Check if your Weaviate URL is correct (should be like 'your-instance.gcp.weaviate.cloud')")
    print("- Verify your Weaviate API key is valid")
    print("- Ensure your Weaviate instance is running")

# Test OpenAI connection
print("\n3. Testing OpenAI connection...")
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=env_vars["OPENAI_API_KEY"])
    
    # Test with a simple embedding
    response = openai_client.embeddings.create(
        input="test",
        model="text-embedding-3-large"
    )
    print("✓ Successfully connected to OpenAI")
    print(f"✓ Embedding dimension: {len(response.data[0].embedding)}")
except Exception as e:
    print(f"✗ OpenAI connection failed: {str(e)}")
    print("\nPossible issues:")
    print("- Check if your OpenAI API key is valid")
    print("- Ensure you have credits in your OpenAI account")

# Check required packages
print("\n4. Checking required packages...")
required_packages = [
    "streamlit",
    "weaviate",
    "openai",
    "unstructured",
    "python-dotenv",
    "tqdm",
    "torch",
]

for package in required_packages:
    try:
        __import__(package.replace("-", "_"))
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} - Not installed")

# Check paths
print("\n5. Checking paths...")
paths_to_check = [
    ("data/documents", "Document storage"),
    ("data/images", "Image storage"),
]

for path, description in paths_to_check:
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
    else:
        print(f"✗ {description}: {path} (will be created)")
        os.makedirs(path, exist_ok=True)

print("\n" + "=" * 50)
print("Debug complete!")
print("=" * 50)