import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the synthetic workforce data
df = pd.read_csv("synthetic_workforce_financials.csv")

# Ensure column names are lowercase and standardised
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Summarise data by staff group
summary = df.groupby("staff_group")[[
    "agency_spend_(£)", "bank_spend_(£)", "substantive_spend_(£)"
]].mean().round(2).to_string()

# Create a prompt for the language model
prompt = f"""
You are a healthcare finance analyst.

Below is average monthly agency, bank and substantive spend by staff group:

{summary}

Please identify trends in spend by staff groups between bank and substantive and suggest possible reasons.
"""

# Load the Mixtral model and tokenizer
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16
)

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate and print the output
response = generator(prompt, max_new_tokens=300, temperature=0.7, do_sample=True)

print("\n🧠 LLM Analysis Output:\n")
print(response[0]["generated_text"])
