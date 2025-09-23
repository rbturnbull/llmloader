import llmloader
from llmloader.openai import OpenAILoader
from langchain_core.messages import AIMessage


def test_azure(azure_mock_setup, credentials):   
    azure_mock, prompt = azure_mock_setup    
    
    llm = llmloader.load(**credentials)   
    azure_mock.assert_called_once()

    credentials["azure_deployment"] = credentials.pop("model")
    credentials["azure_endpoint"] = credentials.pop("api_endpoint")
    azure_mock.assert_called_with(**credentials)

    assert not isinstance(llm, OpenAILoader) # Ensure it's not the OpenAI loader    
    result = llm.invoke(prompt)
    assert isinstance(result, AIMessage)
    assert result.content == prompt  