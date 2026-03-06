import chromadb
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_collection('law_sections')

print(f"Total documents in DB: {collection.count()}")

query_text = 'i am under 18 and driving vehicle and no helmet what is my punishment'
query_embedding = embedder.encode(query_text).tolist()
results = collection.query(query_embeddings=[query_embedding], n_results=10)

for idx, doc in enumerate(results['documents'][0]):
    section = results['metadatas'][0][idx]
    print(f"Section: {section['section']}")
