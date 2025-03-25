=========
llmloader
=========

|pypi|

.. |pypi| image:: https://img.shields.io/pypi/v/llmloader
   :target: https://pypi.org/project/llmloader/

Loads a Langchain LLM by model name as a string.

Installation
============

.. code-block:: bash

    pip install llmloader

Or install from GitHub directly:

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/llmloader.git


Usage
==========

Load the LLM with the `llmloader.load` function. e.g.

.. code-block:: python

    import llmloader

    llm = llmloader.load("gpt-4o")
    result = llm.invoke("Write me a haiku about love")

    llm = llmloader.load("claude-3-5-sonnet-20240620")
    result = llm.invoke("Write me a haiku about love")

    llm = llmloader.load("grok-2-latest")
    result = llm.invoke("Write me a haiku about love")

    llm = llmloader.load("mistral-small-latest")
    result = llm.invoke("Write me a haiku about love")

    llm = llmloader.load("meta-llama/Llama-3.3-70B-Instruct")
    result = llm.invoke("Write me a haiku about love")

CLI
==========

You can test out prompts and models on the command line. Make sure you have your API keys set in your environment or add the key with the ``--api-key`` flag.

.. code-block:: bash
    
    llmloader "Write me a haiku about love" --model gpt-4o-mini
    llmloader "Write me a haiku about love" --model gpt-4o
    llmloader "Write me a haiku about love" --model claude-3-5-sonnet-20240620
    llmloader "Write me a haiku about love" --model grok-2-latest
    llmloader "Write me a haiku about love" --model mistral-small-latest
    llmloader "Write me a haiku about love" --model meta-llama/Meta-Llama-3-8B-Instruct
    llmloader "Write me a haiku about love" --model meta-llama/Llama-3.3-70B-Instruct
    llmloader --help
    

Credit
==========

`Robert Turnbull <https://robturnbull.com>`_  (Melbourne Data Analytics Platform, University of Melbourne)
