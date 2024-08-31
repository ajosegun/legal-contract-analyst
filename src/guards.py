from guardrails import Guard
from guardrails.hub import DetectPII
from guardrails.hub import UnusualPrompt
from openai import AzureOpenAI
from src.config import Config

config = Config()

openai_client = AzureOpenAI(
    azure_deployment=config.AZURE_DEPLOYMENT,
    api_version=config.AZURE_OPENAI_API_VERSION,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
    api_key=config.AZURE_OPENAI_API_KEY,
)


def validate_unusual_prompt(the_prompt: str):
    print("Validating unusual prompt")
    guard = Guard().use(UnusualPrompt, on="prompt", on_fail="exception")

    res = guard(
        openai_client.chat.completions.create,
        prompt=the_prompt,
        model=f"azure/{config.AZURE_DEPLOYMENT}",
    )
    return the_prompt


def validate_pii(prompt: str):
    try:
        guard_pii = Guard().use(
            DetectPII, ["EMAIL_ADDRESS", "PHONE_NUMBER"], "exception"
        )
        guard_pii.validate(prompt)
        return True, prompt
    except Exception as e:
        print(e)
        return False, str(e)
