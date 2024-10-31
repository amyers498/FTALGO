import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define functions for analysis
def load_quickbooks_data(file_path):
    expense_df = pd.read_excel(file_path, sheet_name="Expense Report")
    cash_flow_df = pd.read_excel(file_path, sheet_name="Cash Flow Statement")
    balance_sheet_df = pd.read_excel(file_path, sheet_name="Balance Sheet")
    pnl_df = pd.read_excel(file_path, sheet_name="Profit & Loss Statement")
    return expense_df, cash_flow_df, balance_sheet_df, pnl_df

def select_primary_cost(expense_df):
    expense_summary = expense_df.groupby("Expense_Category")["Amount"].sum().sort_values(ascending=False)
    primary_cost_category = expense_summary.idxmax()
    primary_cost = expense_summary.max()
    return primary_cost_category, primary_cost

def map_expense_to_commodity(expense_category):
    commodity_mapping = {
        "Food Supplies": "Corn",
        "Beverages": "Sugar",
        "Utilities": "Natural Gas",
        "Cleaning Supplies": "Cotton",
        "Packaging": "Wheat"
    }
    return commodity_mapping.get(expense_category, "Corn")

def load_commodity_data(file_path):
    return pd.read_excel(file_path)

def calculate_volatility_and_seasonality(commodity_df):
    commodity_df["Daily_Return"] = commodity_df["Commodity_Price"].pct_change()
    volatility = commodity_df["Daily_Return"].std() * np.sqrt(252)
    commodity_df["Month"] = pd.to_datetime(commodity_df["Date"]).dt.month
    monthly_avg = commodity_df.groupby("Month")["Commodity_Price"].mean()
    high_season = monthly_avg.idxmax() if monthly_avg.max() > 1.5 * monthly_avg.mean() else None
    return volatility * 100, high_season

def assess_financial_profile(cash_flow_df, balance_sheet_df, pnl_df):
    avg_cash_inflow = cash_flow_df["Cash_Inflow"].mean()
    avg_cash_outflow = cash_flow_df["Cash_Outflow"].mean()
    cash_flow_stability = avg_cash_inflow - avg_cash_outflow
    
    avg_assets = balance_sheet_df["Current_Assets"].mean()
    avg_liabilities = balance_sheet_df["Current_Liabilities"].mean()
    current_ratio = avg_assets / avg_liabilities
    
    avg_long_term_debt = balance_sheet_df["Long_Term_Debt"].mean()
    avg_equity = balance_sheet_df["Equity"].mean()
    debt_to_equity_ratio = (avg_liabilities + avg_long_term_debt) / avg_equity
    
    avg_revenue = pnl_df["Revenue"].mean()
    avg_net_income = pnl_df["Net_Income"].mean()
    profit_margin = avg_net_income / avg_revenue
    
    risk_tolerance = "High" if debt_to_equity_ratio > 2 or current_ratio < 1.5 or profit_margin < 0.05 else "Moderate"
    
    financial_profile = {
        "cash_flow_stability": cash_flow_stability,
        "current_ratio": current_ratio,
        "debt_to_equity_ratio": debt_to_equity_ratio,
        "profit_margin": profit_margin,
        "risk_tolerance": risk_tolerance
    }
    return financial_profile

def select_contract_durations(hedge_duration):
    contract_durations = {
        "Short-Term (1-3 months)": [1, 2, 3],
        "Medium-Term (3-12 months)": [3, 6, 12],
        "Long-Term (12+ months)": [12, 18, 24]
    }
    return contract_durations[hedge_duration]

def recommend_hedge_duration(volatility, financial_profile, expense_df):
    monthly_expenses = expense_df.groupby(expense_df['Date'].dt.month)["Amount"].sum()
    high_season = monthly_expenses.idxmax() if monthly_expenses.max() > 1.5 * monthly_expenses.mean() else None
    
    if volatility > 25 or financial_profile["risk_tolerance"] == "High" or high_season:
        hedge_duration = "Short-Term (1-3 months)"
    elif 15 < volatility <= 25 or financial_profile["risk_tolerance"] == "Moderate":
        hedge_duration = "Medium-Term (3-12 months)"
    else:
        hedge_duration = "Long-Term (12+ months)"
    
    return hedge_duration

def recommend_contracts_and_select_best(commodity_df, primary_commodity, hedge_duration, financial_profile):
    commodity_data = commodity_df[commodity_df['Commodity'] == primary_commodity]
    selected_durations = select_contract_durations(hedge_duration)
    contracts = []

    for duration in selected_durations:
        historical_data = commodity_data.tail(duration * 20)
        if historical_data.empty or len(historical_data) < 2:
            continue

        historical_data['Daily_Return'] = historical_data['Commodity_Price'].pct_change()
        historical_return = historical_data['Daily_Return'].mean() * duration * 20
        historical_volatility = historical_data['Daily_Return'].std() * np.sqrt(252)
        
        try:
            recent_trend = historical_data['Commodity_Price'].iloc[-1] / historical_data['Commodity_Price'].iloc[0]
            expected_future_return = historical_return * recent_trend
            expected_future_volatility = historical_volatility * np.sqrt(recent_trend)
            contracts.append({
                "Duration (months)": duration,
                "Historical Average Return (%)": round(historical_return * 100, 2),
                "Historical Volatility (%)": round(historical_volatility * 100, 2),
                "Expected Future Return (%)": round(expected_future_return * 100, 2),
                "Expected Future Volatility (%)": round(expected_future_volatility * 100, 2)
            })
        except IndexError:
            continue

    if contracts:
        best_contract = (min(contracts, key=lambda x: x["Historical Volatility (%)"]) if financial_profile["risk_tolerance"] == "High"
                         else max(contracts, key=lambda x: x["Expected Future Return (%)"] / (x["Expected Future Volatility (%)"] + 1)))
        return best_contract, contracts
    else:
        return None, None

# Streamlit UI
st.title("Hedging Strategy Dashboard")
st.sidebar.header("Upload Files")

quickbooks_file = st.sidebar.file_uploader("Upload QuickBooks Data (Excel)", type=["xlsx"])
commodity_file = st.sidebar.file_uploader("Upload Commodity Options Data (Excel)", type=["xlsx"])

if quickbooks_file and commodity_file:
    # Load data and analyze
    expense_df, cash_flow_df, balance_sheet_df, pnl_df = load_quickbooks_data(quickbooks_file)
    primary_cost_category, primary_cost = select_primary_cost(expense_df)
    mapped_commodity = map_expense_to_commodity(primary_cost_category)
    
    commodity_df = load_commodity_data(commodity_file)
    volatility, high_season = calculate_volatility_and_seasonality(commodity_df[commodity_df["Commodity"] == mapped_commodity])
    financial_profile = assess_financial_profile(cash_flow_df, balance_sheet_df, pnl_df)
    hedge_duration = recommend_hedge_duration(volatility, financial_profile, expense_df)
    best_contract, contracts = recommend_contracts_and_select_best(commodity_df, mapped_commodity, hedge_duration, financial_profile)

    # Display results in a nice format
    st.header("Analysis Results")
    st.subheader("Primary Information")
    st.write(f"**Primary Cost Category**: {primary_cost_category}")
    st.write(f"**Mapped Commodity**: {mapped_commodity}")
    st.write(f"**Financial Profile**: {financial_profile}")
    st.write(f"**Commodity Volatility (%)**: {round(volatility, 2)}")
    st.write(f"**High Season**: {high_season}")
    st.write(f"**Recommended Hedge Duration**: {hedge_duration}")

    if best_contract:
        st.subheader("Best Contract Recommendation")
        st.write(best_contract)

        # Plot historical and expected returns and volatility for visual analysis
        contract_durations = [c["Duration (months)"] for c in contracts]
        historical_returns = [c["Historical Average Return (%)"] for c in contracts]
        historical_volatility = [c["Historical Volatility (%)"] for c in contracts]
        expected_returns = [c["Expected Future Return (%)"] for c in contracts]
        expected_volatility = [c["Expected Future Volatility (%)"] for c in contracts]

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Historical Return vs Volatility
        axs[0].bar(contract_durations, historical_returns, color='blue', label='Return')
        axs[0].plot(contract_durations, historical_volatility, color='red', marker='o', label='Volatility')
        axs[0].set_title("Historical Return and Volatility by Duration")
        axs[0].legend()

        # Expected Future Return vs Volatility
        axs[1].bar(contract_durations, expected_returns, color='blue', label='Expected Return')
        axs[1].plot(contract_durations, expected_volatility, color='red', marker='o', label='Expected Volatility')
        axs[1].set_title("Expected Future Return and Volatility by Duration")
        axs[1].legend()
        axs[1].set_xlabel("Contract Duration (months)")

        # Display the charts in Streamlit
        st.pyplot(fig)

    else:
        st.write("No viable contracts were found based on the given data.")
else:
    st.info("Please upload both QuickBooks data and commodity options data files to proceed.")
