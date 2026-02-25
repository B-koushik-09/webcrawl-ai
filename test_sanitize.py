from modules.llm_generator import LLMGenerator
print('starting')
gen = LLMGenerator('qwen2.5:1.5b')
print('after init model_name=', gen.model_name)
