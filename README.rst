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

To pass an image, it needs to be base64 encoded, and reformatted with LLMWrapper.

.. code-block:: python
    
    import base64
    from langchain_core.messages import HumanMessage
    from llmloader.wrappers import LLMWrapper

    with open("path/to/image.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    formatted_image = LLMWrapper.format(llm, "image", {
        "data": encoded_string,
        "mime_type": "image/png"
    })

    message = HumanMessage(content=[{"type": "text", "text": "Here is an image for you:"}, formatted_image])    

    result = llm.invoke(message)

Get the token usage

.. code-block:: python
        
    from llmloader.wrappers import LLMWrapper

    result = llm.invoke(message)

    count = LLMWrapper.get_token_count(result, record="optional_path/to/record.yaml")


CLI
==========

You can test out prompts and models on the command line. Make sure you have your API keys set in your environment or add the key with the ``--api-key`` flag.

.. code-block:: bash
    
    llmloader "Write me a haiku about love" --model gpt-5-mini
    llmloader "Write me a haiku about love" --model gpt-5.2
    llmloader "Write me a haiku about love" --model claude-sonnet-4-5-20250929
    llmloader "Write me a haiku about love" --model grok-4-latest
    llmloader "Write me a haiku about love" --model mistral-small-latest
    llmloader "Write me a haiku about love" --model gemini-3-pro-preview
    # Using OpenRouter
    llmloader "Write me a haiku about love" --model openai/gpt-5-mini
    # Local deployment models
    llmloader "Write me a haiku about love" --model meta-llama/Meta-Llama-3-8B-Instruct
    llmloader "Write me a haiku about love" --model meta-llama/Llama-3.3-70B-Instruct
    llmloader --help

Environment Variables
======================

You can pass an API key for the model provider using the command line flag ``--api-key``, kwarg ``api_key=...``, or by setting the appropriate environment variable as described below.

================= =========================
Model Provider    Environment Variable
================= =========================
OpenAI            OPENAI_API_KEY
Anthropic         ANTHROPIC_API_KEY
Mistral           MISTRAL_API_KEY
XAI               XAI_API_KEY
OpenRouter        OPENROUTER_API_KEY
Google            GOOGLE_API_KEY
================= =========================

Azure and OpenRouter
------------
To use custom models deployed with Azure OpenAI, you need to set the following environment variables:

- ``CUSTOM_API_KEY``: Your Azure or OpenRouter API key.
- ``CUSTOM_ENDPOINT``: The endpoint URL for your Azure AI or OpenRouter service.

Alternatively, you can pass the endpoint URL directly using the ``--endpoint`` flag.

``--model`` should match the deployment name in your Azure AI resource.

Note: 

- If ``llmloader`` detects the ``OPENAI_API_KEY`` environment variable, it will use the OpenAI API by default if a valid model name is provided and ``CUSTOM_ENDPOINT`` is not set.
- If both ``CUSTOM_API_KEY`` and ``CUSTOM_ENDPOINT`` are set, llmloader will use the Azure or OpenRouter service.
- ``CUSTOM_ENDPOINT`` should be the URL ending with /models, e.g. ``https://your-resource-name.openai.azure.com/models``

Testing
========================

Endpoint Manual Testing
--------------------------

``test_manual.py`` contains tests for models that require API keys. You can run these tests manually after setting the appropriate environment variables.

Once the environment variables are set, you can run the tests with:

.. code-block:: bash

    pytest -m manual

To specify a particular test, use:

.. code-block:: bash

    pytest -m manual tests/test_manual.py::test_name

Credit
==========

- `Robert Turnbull <https://robturnbull.com>`_  (Melbourne Data Analytics Platform, University of Melbourne)
- `James Quang <https://www.linkedin.com/in/jamesquang>`_  (Melbourne Data Analytics Platform, University of Melbourne)
