import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model  # Ensure 'arch' is installed for GARCH model

# Function to load financial data from QuickBooks
def load_quickbooks_data(file_path):
    expense_df = pd.read_excel(file_path, sheet_name="Expense Report")
    cash_flow_df = pd.read_excel(file_path, sheet_name="Cash Flow Statement")
    balance_sheet_df = pd.read_excel(file_path, sheet_name="Balance Sheet")
    pnl_df = pd.read_excel(file_path, sheet_name="Profit & Loss Statement")
    return expense_df, cash_flow_df, balance_sheet_df, pnl_df

# Function to load commodity data from an Excel file
def load_commodity_data(file_path):
    return pd.read_excel(file_path)

# Function to forecast expected returns and volatility using ARIMA and GARCH
def forecast_returns_and_volatility(commodity_data, duration):
    # ARIMA model for expected returns
    model_arima = ARIMA(commodity_data['Commodity_Price'], order=(1, 1, 1))
    arima_fit = model_arima.fit()
    arima_forecast = arima_fit.forecast(steps=duration)
    expected_future_return_monthly = (arima_forecast - commodity_data['Commodity_Price'].iloc[-1]) / commodity_data['Commodity_Price'].iloc[-1]
    
    # GARCH model for expected volatility
    model_garch = arch_model(commodity_data['Daily_Return'].dropna(), vol='Garch', p=1, q=1)
    garch_fit = model_garch.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=duration)
    expected_future_volatility = np.sqrt(garch_forecast.variance.values[-1, :]) * 100

    return expected_future_return_monthly * 100, expected_future_volatility

# Function to recommend the best contracts
def recommend_contracts_and_select_best(commodity_df, selected_commodity, hedge_duration):
    commodity_data = commodity_df[commodity_df['Commodity'] == selected_commodity]
    commodity_data["Daily_Return"] = commodity_data["Commodity_Price"].pct_change()
    selected_durations = select_contract_durations(hedge_duration)
    contracts = []

    for duration in selected_durations:
        historical_data = commodity_data.tail(duration * 20)
        if historical_data.empty or len(historical_data) < 2:
            continue

        # Forecast expected returns and volatility for each duration
        expected_future_return_monthly, expected_future_volatility = forecast_returns_and_volatility(historical_data, duration)

        # Filter for contracts with positive returns
        if expected_future_return_monthly.mean() > 0:
            contracts.append({
                "Duration (months)": duration,
                "Expected Future Monthly Return (%)": expected_future_return_monthly.mean(),
                "Expected Future Volatility (%)": expected_future_volatility.mean(),
                "Expected Monthly Returns": expected_future_return_monthly.tolist()
            })

    # Sort by expected return and select top 3
    contracts = sorted(contracts, key=lambda x: x["Expected Future Monthly Return (%)"], reverse=True)[:3]
    return contracts

# Streamlit UI setup
st.title("Enhanced Hedging Strategy Dashboard")

# Sidebar file uploads
quickbooks_file = st.sidebar.file_uploader("Upload QuickBooks Data (Excel)", type=["xlsx"])
commodity_file = st.sidebar.file_uploader("Upload Commodity Options Data (Excel)", type=["xlsx"])

if quickbooks_file and commodity_file:
    # Load data
    expense_df, cash_flow_df, balance_sheet_df, pnl_df = load_quickbooks_data(quickbooks_file)
    commodity_df = load_commodity_data(commodity_file)

    # Define primary cost and correlated commodity selection methods
    primary_cost_category, primary_cost = select_primary_cost(expense_df)
    selected_commodity = auto_select_commodity(expense_df, commodity_df)

    # Assess financial profile and volatility
    financial_profile = assess_financial_profile(cash_flow_df, balance_sheet_df, pnl_df)
    hedge_duration = recommend_hedge_duration(volatility, financial_profile, expense_df)
    top_contracts = recommend_contracts_and_select_best(commodity_df, selected_commodity, hedge_duration)

    # Display top 3 contract recommendations with charts
    if top_contracts:
        st.subheader("Top 3 Contract Recommendations")

        for idx, contract in enumerate(top_contracts, start=1):
            st.write(f"### Contract {idx}")
            st.write(f"**Duration (months):** {contract['Duration (months)']}")
            st.write(f"**Expected Future Monthly Return (%):** {contract['Expected Future Monthly Return (%)']:.2f}")
            st.write(f"**Expected Future Volatility (%):** {contract['Expected Future Volatility (%)']:.2f}")

            # Expected monthly returns for the duration of the contract
            expected_monthly_returns = contract['Expected Monthly Returns']
            months = list(range(1, contract['Duration (months)'] + 1))

            # Plot expected returns
            sns.set_style("whitegrid")
            fig, ax1 = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=months, y=expected_monthly_returns, marker='o', ax=ax1)
            ax1.set_title(f"Expected Monthly Returns for Contract {idx}")
            ax1.set_xlabel("Month")
            ax1.set_ylabel("Expected Return (%)")
            st.pyplot(fig)

            # Plot volatility over the duration
            fig, ax2 = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=months, y=[contract["Expected Future Volatility (%)"]] * len(months), marker='o', color='red', ax=ax2)
            ax2.set_title(f"Expected Volatility for Contract {idx}")
            ax2.set_xlabel("Month")
            ax2.set_ylabel("Volatility (%)")
            st.pyplot(fig)

    else:
        st.write("No viable contracts with positive expected returns were found.")
else:
    st.info("Please upload both QuickBooks data and commodity options data files to proceed.")
