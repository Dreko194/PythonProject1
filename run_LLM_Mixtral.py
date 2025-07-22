import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# Load HF token from environment (do NOT hardcode it)
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Missing Hugging Face token. Set HF_TOKEN as environment variable.")

# Authenticate with Hugging Face
login(token=hf_token)

# Load the synthetic workforce data
df = pd.read_csv("synthetic_workforce_financials.csv")

# Standardise column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Create a summary table of average monthly spend by staff group
summary = df.groupby("staff_group")[[
    "agency_spend_(£)", "bank_spend_(£)", "substantive_spend_(£)"
]].mean().round(2).to_string()

# Define the prompt for the language model
prompt = f"""
You are a healthcare finance analyst.

Below is the average monthly agency, bank, and substantive spend by staff group:

{summary}

Please identify trends in spend between bank and substantive for different staff groups, and suggest possible reasons.
"""

# Load the Mixtral model and tokenizer
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16, use_auth_token=hf_token
)

# Create text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate a response
response = generator(prompt, max_new_tokens=300, temperature=0.7, do_sample=True)

# Output the result
print("\n🧠 LLM Analysis Output:\n")
print(response[0]["generated_text"])
