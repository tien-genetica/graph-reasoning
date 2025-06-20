from sentence_transformers import SentenceTransformer

# It's good practice to use the same model as your main script
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

# The terms you want to find paths between
term1 = "ambroxol"
term2 = "GCase"

# Generate embeddings
embedding1 = model.encode([term1])[0].tolist()
embedding2 = model.encode([term2])[0].tolist()

print(f"Embedding for '{term1}':\\n", embedding1)
print("\\n" + "="*40 + "\\n")
print(f"Embedding for '{term2}':\\n", embedding2) 