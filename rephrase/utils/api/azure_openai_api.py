from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from openai import AzureOpenAI


def get_azure_openai_client(api_version: str, endpoint: str):
    # Setup Microsoft EntraID for auth : https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/managed-identity
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
    )
    return client