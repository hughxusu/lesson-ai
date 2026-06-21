from transformers import GPT2Tokenizer, OPTForCausalLM

model_id = "facebook/opt-6.7b"

model = OPTForCausalLM.from_pretrained(model_id, load_in_8bit=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_id)