=========
llmloader
=========

Loads a Langchain LLM by model name as a string.


Usage
==========

Load the LLM with the `llmloader.load` function. e.g.

.. code-block:: python

    import llmloader

    llm = llmloader.load("gpt-4o")
    result = llm.invoke("Write me a haiku about love")

    llm = llmloader.load("claude-3-5-sonnet-20240620")
    result = llm.invoke("Write me a haiku about love")

    llm = llmloader.load("meta-llama/Llama-3.3-70B-Instruct")
    result = llm.invoke("Write me a haiku about love")

CLI
==========

You can test out prompts and models on the command line.

.. code-block:: bash
    
    llmloader "Write me a haiku about love" --model gpt-4o-mini
    llmloader "Write me a haiku about love" --model gpt-4o
    llmloader "Write me a haiku about love" --model claude-3-5-sonnet-20240620
    llmloader "Write me a haiku about love" --model meta-llama/Meta-Llama-3-8B-Instruct
    llmloader "Write me a haiku about love" --model meta-llama/Llama-3.3-70B-Instruct
    llmloader --help
    

Credit
==========

Robert Turnbull (Melbourne Data Analytics Platform, University of Melbourne)
