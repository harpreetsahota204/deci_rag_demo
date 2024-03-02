import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import intel_extension_for_pytorch as ipex
model_name = "Deci/DeciLM-7B"
device = "xpu" # for GPU usage or "cpu" for CPU usage
 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True).to(device)
 
inputs = tokenizer.encode("In a shocking finding, scientists discovered a herd of unicorns living in", return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_p=0.95)
print(tokenizer.decode(outputs[0]))
 
# The model can also be used via the text-generation pipeline interface
from transformers import pipeline
generator = pipeline("text-generation", "Deci/DeciLM-7B", torch_dtype="auto", trust_remote_code=True, device=device)
outputs = generator("In a shocking finding, scientists discovered a herd of unicorns living in", max_new_tokens=100, do_sample=True, top_p=0.95)
print(outputs[0]["generated_text"])
