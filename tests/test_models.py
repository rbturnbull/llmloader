import llmloader
from llmloader.openai import OpenAILoader
from langchain_core.messages import AIMessage


def test_azure(azure_mock_setup, credentials_azure):   
    azure_mock, prompt = azure_mock_setup    
    
    llm = llmloader.load(**credentials_azure)   
    azure_mock.assert_called_once()

    credentials_azure["azure_deployment"] = credentials_azure.pop("model")
    credentials_azure["azure_endpoint"] = credentials_azure.pop("api_endpoint")
    azure_mock.assert_called_with(**credentials_azure)

    assert not isinstance(llm, OpenAILoader) # Ensure it's not the OpenAI loader    
    result = llm.invoke(prompt)
    assert isinstance(result, AIMessage)
    assert result.content == prompt  

def test_gemini(gemini_mock_setup, credentials_gemini):
    gemini_mock, prompt = gemini_mock_setup

    llm = llmloader.load(**credentials_gemini)
    gemini_mock.assert_called_once()    
    result = llm.invoke(prompt)
    assert isinstance(result, AIMessage)
    assert result.content == prompt

def test_openrouter(openrouter_mock_setup, credentials_openrouter):
    openrouter_mock, prompt = openrouter_mock_setup

    llm = llmloader.load(**credentials_openrouter)
    openrouter_mock.assert_called_once()    
    result = llm.invoke(prompt)
    assert isinstance(result, AIMessage)
    assert result.content == prompt