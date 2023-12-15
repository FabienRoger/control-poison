from lowstakes.llm import OpenAIChatModel, gpt_3_5, gpt_4, call_llm

TRUSTED_MODEL = gpt_3_5()
UNTRUSTED_MODEL = gpt_4()
GOLD_LABELER = gpt_4()