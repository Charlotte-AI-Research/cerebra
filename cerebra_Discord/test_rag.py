from rag import RAGPipeline

def test_rag():
    print("Initializing RAG Pipeline for testing...")
    rag = RAGPipeline()
    
    print("\n--- Test Query 1: Club Summary ---")
    query1 = "What is the mission of CAIR?"
    print(f"Query: {query1}")
    response1 = rag.query(query1)
    print(f"Response: {response1}")

    print("\n--- Test Query 2: Announcements ---")
    query2 = "When is voting happening?"
    print(f"Query: {query2}")
    response2 = rag.query(query2)
    print(f"Response: {response2}")

if __name__ == "__main__":
    test_rag()
