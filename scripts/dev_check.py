try:
    import langchain
    import llama_index
    import tiktoken
    import numpy
    import pandas
    import sklearn
    print("Imports OK — environment set up correctly!")
except Exception as e:
    print("Import error:", e)
