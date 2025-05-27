import chromadb
from pprint import pprint 
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name='my-collection')
collection.add(
    documents=[
        "this is a document about pineapples",
        "this is a document about apples",
        "this is a document about oranges"
    ], ids=[
        'id1',
        'id2',
        'id3'
    ]
)


results = collection.query(
    query_texts=['this is about hawai'],
    n_results = 10
)

pprint(results)
print("--- Semantic Similarity ---")
pprint([1/x for x in results['distances'][0]])