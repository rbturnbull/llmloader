import pytest

pytest_plugins = ["mocks.models"]

@pytest.fixture()
def credentials(model_auth):
    """Pytest fixture providing complete credential configurations for all models.

    Takes the model authentication configurations and enriches them with common
    base credentials (api_key, temperature, max_tokens) that are shared across
    all model providers for testing purposes.

    Args:
        model_auth (dict): The model_auth fixture containing basic model
            configurations for each provider.

    Returns:
        dict: A dictionary mapping model provider names to their complete
            credential dictionaries, including both provider-specific settings
            and common base credentials.
    """
    base_credentials = {
        "api_key": "key123",
        "temperature": 0.7,
        "max_tokens": 100,
    }
    for authdata in model_auth.values():
        authdata.update(base_credentials)
    return model_auth