import pandas as pd

df_workforce = pd.read_csv("synthetic_workforce_financials.csv")
df_costing = pd.read_csv("synthetic_nhs_standard_costing_return.csv")
df_balance = pd.read_csv("synthetic_balance_sheet.csv")
df_monitoring = pd.read_csv("synthetic_financial_monitoring.csv")
df_slr = pd.read_csv("synthetic_slr.csv")
df_reference = pd.read_csv("synthetic_reference_costs_plics.csv")

# Example: print top 5 rows of each
print("Workforce Financials:\n", df_workforce.head())
print("Costing Return:\n", df_costing.head())
print("Balance Sheet:\n", df_balance.head())
print("Financial Monitoring:\n", df_monitoring.head())
print("SLR Data:\n", df_slr.head())
print("Reference Costs / PLICS:\n", df_reference.head())
