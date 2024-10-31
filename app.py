import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import seaborn as sns

# Load QuickBooks Data
def load_quickbooks_data(file):
    try:
        expense_df = pd.read_excel(file, sheet_name="Expense Report")
        cash_flow_df = pd.read_excel(file, sheet_name="Cash Flow Statement")
        balance_sheet_df = pd.read_excel(file, sheet_name="Balance Sheet")
        pnl_df = pd.read_excel(file, sheet_name="Profit & Loss Statement")
        return expense_df, cash_flow_df, balance_sheet_df, pnl_df
    except Exception:
        st.error("Error loading QuickBooks data.")
        st.stop()

# Load Commodity Data
def load_commodity_data(file):
    try:
        return pd.read_excel(file)
    except Exception:
        st.error("Error loading Commodity data.")
        st.stop()

# Function to select primary expense category
def select_primary_cost(expense_df, manual_category=None):
    if manual_category:
        primary_cost_category = manual_category
        primary_cost = expense_df[expense_df["Expense_Category"] == manual_category]["Amount"].sum()
    else:
        expense_summary = expense_df.groupby("Expense_Category")["Amount"].sum().sort_values(ascending=False)
        primary_cost_category = expense_summary.idxmax()
        primary_cost = expense_summary.max()
    return primary_cost_category, primary_cost

# Forecast Returns & Volatility
def forecast_returns_and_volatility(commodity_data, duration):
    model_arima = ARIMA(commodity_data['Commodity_Price'], order=(1, 1, 1))
    arima_fit = model_arima.fit()
    arima_forecast = arima_fit.forecast(steps=duration)
    expected_monthly_return = ((arima_forecast - commodity_data['Commodity_Price'].iloc[-1]) / commodity_data['Commodity_Price'].iloc[-1]) * 100

    model_garch = arch_model(commodity_data['Daily_Return'].dropna(), vol='Garch', p=1, q=1)
    garch_fit = model_garch.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=duration)
    expected_volatility = np.sqrt(garch_forecast.variance.values[-1, :]) * 100

    return expected_monthly_return, expected_volatility

# Recommend Contracts
def recommend_contracts_and_select_best(commodity_df, selected_commodity, hedge_duration):
    commodity_data = commodity_df[commodity_df['Commodity'] == selected_commodity]
    commodity_data["Daily_Return"] = commodity_data["Commodity_Price"].pct_change()
    contract_durations = {"Short-Term": [1, 2, 3], "Medium-Term": [3, 6, 12], "Long-Term": [12, 18, 24]}
    durations = contract_durations[hedge_duration]
    contracts = []

    for duration in durations:
        historical_data = commodity_data.tail(duration * 20)
        if historical_data.empty or len(historical_data) < 2:
            continue

        monthly_returns, volatility = forecast_returns_and_volatility(historical_data, duration)
        if monthly_returns.mean() > 0:  # Only consider positive expected returns
            contracts.append({
                "Duration (months)": duration,
                "Expected Monthly Return (%)": monthly_returns.mean(),
                "Expected Volatility (%)": volatility.mean(),
                "Monthly Returns": monthly_returns.tolist()
            })

    return sorted(contracts, key=lambda x: x["Expected Monthly Return (%)"], reverse=True)[:3]

# Streamlit Setup
st.title("Enhanced Hedging Strategy Dashboard")
quickbooks_file = st.sidebar.file_uploader("Upload QuickBooks Data (Excel)", type=["xlsx"])
commodity_file = st.sidebar.file_uploader("Upload Commodity Options Data (Excel)", type=["xlsx"])

if quickbooks_file and commodity_file:
    expense_df, cash_flow_df, balance_sheet_df, pnl_df = load_quickbooks_data(quickbooks_file)
    commodity_df = load_commodity_data(commodity_file)

    # Select primary cost category
    primary_cost_category, primary_cost = select_primary_cost(expense_df)

    # Set example values for selected commodity and hedge duration
    selected_commodity = "Auto-selected commodity"  # Example placeholder
    hedge_duration = "Medium-Term"  # Example placeholder

    # Display Top Contracts
    top_contracts = recommend_contracts_and_select_best(commodity_df, selected_commodity, hedge_duration)
    if top_contracts:
        st.subheader("Top 3 Contract Recommendations")
        sns.set_theme(style="whitegrid")

        for idx, contract in enumerate(top_contracts, start=1):
            st.write(f"### Contract {idx}")
            st.write(f"**Duration (months):** {contract['Duration (months)']}")
            st.write(f"**Expected Monthly Return (%):** {contract['Expected Monthly Return (%)']:.2f}")
            st.write(f"**Expected Volatility (%):** {contract['Expected Volatility (%)']:.2f}")

            # Plot Expected Monthly Returns
            fig, ax1 = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=range(1, contract['Duration (months)'] + 1), y=contract['Monthly Returns'], ax=ax1, marker="o")
            ax1.set_title(f"Expected Monthly Returns for Contract {idx}")
            ax1.set_xlabel("Month")
            ax1.set_ylabel("Expected Return (%)")
            st.pyplot(fig)

            # Plot Expected Volatility
            fig, ax2 = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=range(1, contract['Duration (months)'] + 1), y=[contract['Expected Volatility (%)']] * contract['Duration (months)'], ax=ax2, marker="o", color="red")
            ax2.set_title(f"Expected Volatility for Contract {idx}")
            ax2.set_xlabel("Month")
            ax2.set_ylabel("Volatility (%)")
            st.pyplot(fig)
    else:
        st.write("No contracts with positive returns found.")
else:
    st.info("Please upload both QuickBooks data and commodity options data files to proceed.")
