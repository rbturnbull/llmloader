=========
llmloader
=========

.. start-badges

|pypi| |testing badge| |black badge|

.. |pypi| image:: https://img.shields.io/pypi/v/llmloader?color=blue
   :target: https://pypi.org/project/llmloader/

.. |testing badge| image:: https://github.com/rbturnbull/llmloader/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/llmloader/actions

.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. end-badges   

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

    llm = llmloader.load("grok-4-latest")
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
    llmloader "Write me a haiku about love" --model grok-4-latest
    llmloader "Write me a haiku about love" --model mistral-small-latest
    llmloader "Write me a haiku about love" --model meta-llama/Meta-Llama-3-8B-Instruct
    llmloader "Write me a haiku about love" --model meta-llama/Llama-3.3-70B-Instruct
    llmloader --help

Environment Variables
======================
To use custom models deployed with Azure OpenAI, you need to set the following environment variables:

- AZURE_OPENAI_API_KEY: Your Azure OpenAI API key.
- AZURE_OPENAI_API_VERSION: The API version to use (e.g., "2024-02-15-preview").
- AZURE_OPENAI_ENDPOINT: The endpoint URL for your Azure OpenAI service.

``--model`` should match the deployment name in your Azure OpenAI resource.

Note: If ``llmloader`` detects the OPENAI_API_KEY environment variable, it will use the OpenAI API by default if a valid model name is provided.
    

Credit
==========

`Robert Turnbull <https://robturnbull.com>`_  (Melbourne Data Analytics Platform, University of Melbourne)
