from rag import RAGPipeline

def test_rag():
    print("Initializing RAG Pipeline for testing...")
    rag = RAGPipeline()
    
    print("\n--- Test Query 1: Club Summary ---")
    query1 = "What is the mission of CAIR?"
    print(f"Query: {query1}")
    response1 = rag.query(query1)
    print(f"Response: {response1}")

    print("\n--- Test Query 2: Departments ---")
    query2 = "What are the departments of majors under University of North Carolina at Charlotte?"
    print(f"Query: {query2}")
    response2 = rag.query(query2)
    print(f"Response: {response2}")

    print("\n--- Test Query 3: Courses ---")
    query3 = "What are all 5000 level courses in CCI?"
    print(f"Query: {query3}")
    response3 = rag.query(query3)
    print(f"Response: {response3}")

if __name__ == "__main__":
    test_rag()
