import os

from azure.core.exceptions import ClientAuthenticationError
from langchain_openai import AzureChatOpenAI
from azure.identity import InteractiveBrowserCredential, get_bearer_token_provider

# --- Azure OpenAI (LangChain) Configuration ---
def get_azure_openai_client():
    try:
        azure_scope = os.getenv("AZURE_OPENAI_SCOPE")
        # Use only InteractiveBrowserCredential for local dev
        credential = InteractiveBrowserCredential()
        token_provider = get_bearer_token_provider(credential, azure_scope)

        AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        azure_client = AzureChatOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            azure_ad_token_provider=token_provider,
            api_version=AZURE_API_VERSION,
            deployment_name=AZURE_DEPLOYMENT,
            timeout=120,  # 120 seconds for vision processing
            max_retries=2,
        )
        return azure_client
    except Exception as e:
        print(f"ERROR: Failed to configure Azure OpenAI client: {e}")
        return None
