import pytest
from langchain_core.messages import AIMessage   
from unittest.mock import MagicMock, patch

@pytest.fixture()
def azure_mock_setup():
    prompt = "Write me a haiku about love"
    with patch("llmloader.azure.AzureChatOpenAI") as azure_mock:
        azure_mock_client = MagicMock()
        azure_mock.return_value = azure_mock_client
        azure_mock_client.invoke.return_value = AIMessage(content=prompt)      
        yield azure_mock, prompt

@pytest.fixture()
def credentials():
    return {
        "model": "deployed_model_name",
        "temperature": 0.7,
        "api_key": "dummykey123",
        "max_tokens": 100,
        "api_version": "api_version",
        "api_endpoint": "api_endpoint",       
    }  