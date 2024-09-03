from guardrails import Guard, OnFailAction
from guardrails.hub import DetectPII, UnusualPrompt, ToxicLanguage
from openai import AzureOpenAI
from src.config import Config

config = Config()

openai_client = AzureOpenAI(
    azure_deployment=config.AZURE_DEPLOYMENT,
    api_version=config.AZURE_OPENAI_API_VERSION,
    azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
    api_key=config.AZURE_OPENAI_API_KEY,
)


def validate_input(the_prompt: str):
    print("Validating input")
    guard = Guard().use_many(
        ToxicLanguage(
            threshold=0.5, validation_method="sentence", on_fail=OnFailAction.EXCEPTION
        ),
        DetectPII(["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail=OnFailAction.EXCEPTION),
        # UnusualPrompt(on="prompt", on_fail="exception"),
    )

    try:
        guard.validate(the_prompt)
        return True, the_prompt
    except Exception as e:
        print(e)
        return False, str(e)


# def validate_unusual_prompt(the_prompt: str):
#     print("Validating unusual prompt")
#     guard = Guard().use(UnusualPrompt, on="prompt", on_fail="exception")

#     res = guard(
#         openai_client.chat.completions.create,
#         prompt=the_prompt,
#         model=f"azure/{config.AZURE_DEPLOYMENT}",
#     )
#     return the_prompt
