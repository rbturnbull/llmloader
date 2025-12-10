import pytest, warnings
from langchain_core.messages import AIMessage   
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

@pytest.fixture()
def prompt():
    return "Write me a haiku about love"

@pytest.fixture()
def azure_mock_setup(prompt):
    with patch("llmloader.azure.AzureChatOpenAI") as azure_mock:
        prompt = str(prompt)
        azure_mock_client = MagicMock()
        azure_mock.return_value = azure_mock_client
        azure_mock_client.invoke.return_value = AIMessage(content=prompt)      
        yield azure_mock, prompt

@pytest.fixture()
def gemini_mock_setup(prompt):
    with patch("llmloader.gemini.ChatGoogleGenerativeAI") as gemini_mock:
        prompt = str(prompt)
        gemini_mock_client = MagicMock()
        gemini_mock.return_value = gemini_mock_client
        gemini_mock_client.invoke.return_value = AIMessage(content=prompt)
        yield gemini_mock, prompt

@pytest.fixture()
def openrouter_mock_setup(prompt):
    with patch("llmloader.openrouter.ChatOpenAI") as openrouter_mock:
        prompt = str(prompt)
        openrouter_mock_client = MagicMock()
        openrouter_mock.return_value = openrouter_mock_client
        openrouter_mock_client.invoke.return_value = AIMessage(content=prompt)      
        yield openrouter_mock, prompt

@pytest.fixture()
def force_azure_by_fail_openai(prompt):    
    prompt = str(prompt)
    # Mock OpenAILoader's __call__ method to return None, forcing fallback to Azure
    with patch("llmloader.openai.OpenAILoader.__call__") as openai_loader_call:
        openai_loader_call.return_value = None
        # Mock AzureChatOpenAI to return a response without actually calling Azure
        with patch("llmloader.azure.AzureChatOpenAI") as azure_mock:
            azure_mock_client = MagicMock()
            azure_mock.return_value = azure_mock_client
            azure_mock_client.invoke.return_value = AIMessage(content=prompt)                  
            yield prompt

@pytest.fixture()
def credentials_azure():
    return {
        "model": "deployed_model_name",
        "temperature": 0.7,
        "api_key": "dummykey123",
        "max_tokens": 100,
        "api_version": "api_version",
        "api_endpoint": "api_endpoint",       
    }  

@pytest.fixture()
def credentials_gemini():
    return {
        "model": "gemini-2.5-flash",
        "temperature": 0.7,
        "api_key": "dummykey456",
        "max_tokens": 100,        
    }

@pytest.fixture()
def credentials_openrouter():
    return {
        "model": "openrouter-model-name",
        "temperature": 0.7,
        "api_key": "dummykey789",
        "max_tokens": 100,
    }   