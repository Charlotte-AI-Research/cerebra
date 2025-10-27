try:
    import langchain, llama_index, tiktoken
    import numpy, pandas, sklearn
    print("Imports OK — environment set up correctly!")
except Exception as e:
    print("Import error:", e)
