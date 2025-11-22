"""Pre-download sentence transformer model during build."""
from sentence_transformers import SentenceTransformer

print("Downloading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✓ Model cached successfully")
