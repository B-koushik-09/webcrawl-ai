import sys
sys.path.append('.')
from modules.embeddings import get_knowledge_index

index = get_knowledge_index()
collection = index.collection

results = collection.get(
    where={"dept": "CSE-AIML"}
)

print(f"\n=========================================")
print(f"Total chunks with dept=CSE-AIML: {len(results['ids'])}")

found = False
for i, content in enumerate(results['documents']):
    if "Sagar Yeruva" in content or "HOD" in content:
        print(f"\n--- MATCH FOUND AT INDEX {i} ---")
        print(f"ID: {results['ids'][i]}")
        print(f"Metadata: {results['metadatas'][i]}")
        print(f"Content length: {len(content)}")
        print("-------------")
        print(content[:300])
        found = True

if not found:
    print("\nCRITICAL FAILURE: Neither 'Sagar Yeruva' nor 'HOD' found in ANY CSE-AIML chunk!")
    print("Let me print the first 3 chunks to see what's actually there:")
    for i in range(min(3, len(results['ids']))):
        print(f"\n[{i}] {results['documents'][i][:200]}...")

print(f"=========================================\n")
