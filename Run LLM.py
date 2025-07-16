import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the synthetic workforce data
df = pd.read_csv("synthetic_workforce_financials.csv")

# Ensure column names are lowercase and standardised
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Summarise data by staff group
summary = df.groupby("staff_group")[["agency_spend_(£)", "bank_spend_(£)", "substantive_spend_(£)"]].mean().round(2).to_string()

# Create a prompt for the language model
prompt = f"""
You are a healthcare finance analyst.

Below is average monthly agency, bank and substantive spend by staff group:

{summary}

Please identify trends in spend by staff groups between bank and substantive suggest possible reasons.
"""

# Load the Qwen model and tokenizer
model_id = "Qwen/Qwen1.5-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

# Encode input and generate output
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150)

# Decode and print the model’s response
print("\n🧠 LLM Analysis Output:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
