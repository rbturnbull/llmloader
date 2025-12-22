def call_model_and_assert(name, mock_setup, credentials, supp_credentials=None):    
    """Test helper function to load a model and verify its behavior.

    Loads a model using llmloader, verifies the mock was called with expected
    credentials, invokes the model with a prompt, and asserts the response
    is an AIMessage with the expected content.

    Args:
        name (str): The name of the model to test (e.g., 'openai', 'anthropic').
        mock_setup (tuple): A tuple containing (mock_object, prompt_string).
        credentials (dict): Dictionary mapping model names to their credential dictionaries.
        supp_credentials (dict, optional): Supplementary credentials to verify the mock
            was called with. If None, uses credentials[name] for verification.

    Raises:
        AssertionError: If the mock wasn't called correctly or if the model response
            doesn't match expectations.
        Exception: Any exception raised during model loading or invocation.
    """    
    try:
        mock, prompt = mock_setup
        import llmloader
        llm = llmloader.load(**credentials[name])           
        if supp_credentials:
            mock.assert_called_once_with(**supp_credentials)
        else:
            mock.assert_called_once_with(**credentials[name])
        result = llm.invoke(prompt)
        from langchain_core.messages import AIMessage
        assert isinstance(result, AIMessage)
        assert result.content == prompt
    except Exception as e:
        raise e

def test_openai(openai_mock_setup, credentials):
    """Test OpenAI model loading and invocation.

    Args:
        openai_mock_setup (tuple): Fixture providing mocked OpenAI setup.
        credentials (dict): Fixture providing test credentials for all models.
    """
    call_model_and_assert("openai", openai_mock_setup, credentials)

def test_anthropic(anthropic_mock_setup, credentials):
    """Test Anthropic model loading and invocation.

    Args:
        anthropic_mock_setup (tuple): Fixture providing mocked Anthropic setup.
        credentials (dict): Fixture providing test credentials for all models.
    """
    call_model_and_assert("anthropic", anthropic_mock_setup, credentials)

def test_gemini(gemini_mock_setup, credentials):
    """Test Gemini model loading and invocation.

    Args:
        gemini_mock_setup (tuple): Fixture providing mocked Gemini setup.
        credentials (dict): Fixture providing test credentials for all models.
    """
    call_model_and_assert("gemini", gemini_mock_setup, credentials)        

def test_xai(xai_mock_setup, credentials):
    """Test XAI model loading and invocation.

    Args:
        xai_mock_setup (tuple): Fixture providing mocked XAI setup.
        credentials (dict): Fixture providing test credentials for all models.
    """
    call_model_and_assert("xai", xai_mock_setup, credentials)

def test_mistral(mistral_mock_setup, credentials):
    """Test Mistral model loading and invocation.

    Args:
        mistral_mock_setup (tuple): Fixture providing mocked Mistral setup.
        credentials (dict): Fixture providing test credentials for all models.
    """
    call_model_and_assert("mistral", mistral_mock_setup, credentials)

def test_llama(llama_mock_setup, credentials):
    """Test Llama model loading and invocation with custom credential mapping.

    Llama requires special handling where the 'model' parameter is renamed to 'llm'
    and certain parameters (api_key, temperature, max_tokens) are removed before
    verifying the mock call.

    Args:
        llama_mock_setup (tuple): Fixture providing mocked Llama setup.
        credentials (dict): Fixture providing test credentials for all models.
    """
    supp_credentials = credentials["llama"].copy()
    supp_credentials['llm'] = supp_credentials.pop('model', None)
    supp_credentials.pop('api_key', None)
    supp_credentials.pop('temperature', None)
    supp_credentials.pop('max_tokens', None)
    call_model_and_assert("llama", llama_mock_setup, credentials, supp_credentials)

def test_azure(azure_mock_setup, credentials, monkeypatch):
    """Test Azure model loading and invocation with environment variable configuration.

    Azure requires special handling where the endpoint is set via environment variable,
    and the 'api_key' parameter is renamed to 'credential' before verifying the mock call.

    Args:
        azure_mock_setup (tuple): Fixture providing mocked Azure setup.
        credentials (dict): Fixture providing test credentials for all models.
        monkeypatch: Pytest fixture for modifying environment variables.
    """
    monkeypatch.setenv("CUSTOM_ENDPOINT", "https://dummy-azure-endpoint.open")
    supp_credentials = credentials["azure"].copy()
    from os import getenv
    supp_credentials['endpoint'] = getenv("CUSTOM_ENDPOINT", "")
    supp_credentials['credential'] = supp_credentials.pop('api_key', "")
    call_model_and_assert("azure", azure_mock_setup, credentials, supp_credentials)

def test_openrouter(openrouter_mock_setup, credentials):
    """Test OpenRouter model loading and invocation with custom base URL.

    OpenRouter requires the base_url to be set to the OpenRouter API endpoint
    before verifying the mock call.

    Args:
        openrouter_mock_setup (tuple): Fixture providing mocked OpenRouter setup.
        credentials (dict): Fixture providing test credentials for all models.
    """
    supp_credentials = credentials["openrouter"].copy()
    supp_credentials['base_url'] = "https://openrouter.ai/api/v1"
    call_model_and_assert("openrouter", openrouter_mock_setup, credentials, supp_credentials)        
