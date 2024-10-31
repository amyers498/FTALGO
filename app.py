import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model  # Ensure 'arch' is installed for GARCH model
import seaborn as sns 

# Function to load financial data from QuickBooks
def load_quickbooks_data(file_path):
    # Load each sheet from QuickBooks Excel file into separate dataframes
    expense_df = pd.read_excel(file_path, sheet_name="Expense Report")
    cash_flow_df = pd.read_excel(file_path, sheet_name="Cash Flow Statement")
    balance_sheet_df = pd.read_excel(file_path, sheet_name="Balance Sheet")
    pnl_df = pd.read_excel(file_path, sheet_name="Profit & Loss Statement")
    return expense_df, cash_flow_df, balance_sheet_df, pnl_df  # Return all dataframes

# Function to correlate each expense category with each commodity
def correlate_expenses_with_commodities(expense_df, commodity_df):
    expense_categories = expense_df["Expense_Category"].unique()
    correlations = []

    for category in expense_categories:
        category_expense = expense_df[expense_df["Expense_Category"] == category]
        category_monthly_totals = category_expense.groupby(category_expense['Date'].dt.to_period("M"))["Amount"].sum()

        for commodity in commodity_df["Commodity"].unique():
            commodity_prices = commodity_df[commodity_df["Commodity"] == commodity].set_index("Date")["Commodity_Price"]
            commodity_prices = commodity_prices.resample("M").mean()
            combined = pd.DataFrame({"Expense": category_monthly_totals, "Commodity": commodity_prices}).dropna()
            correlation = combined["Expense"].corr(combined["Commodity"])

            correlations.append({
                "Expense_Category": category,
                "Commodity": commodity,
                "Correlation": correlation
            })

    correlations_df = pd.DataFrame(correlations).sort_values(by="Correlation", ascending=False)
    return correlations_df[correlations_df["Correlation"] > 0]  # Only positive correlations

# Improved function to predict returns and volatility using ARIMA and GARCH models
def forecast_returns_and_volatility(commodity_data, duration):
    # ARIMA model to predict return
    model_arima = ARIMA(commodity_data['Commodity_Price'], order=(1, 1, 1))
    arima_fit = model_arima.fit()
    arima_forecast = arima_fit.forecast(steps=duration)
    expected_future_return = arima_forecast.mean() / commodity_data['Commodity_Price'].iloc[-1] - 1

    # GARCH model to predict volatility
    model_garch = arch_model(commodity_data['Daily_Return'].dropna(), vol='Garch', p=1, q=1)
    garch_fit = model_garch.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=duration)
    expected_future_volatility = np.sqrt(garch_forecast.variance.values[-1, :]).mean() * 100  # Annualized volatility

    return expected_future_return * 100, expected_future_volatility  # Return percentage values

# Assess company's financial health and risk tolerance
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
    
    return {
        "cash_flow_stability": cash_flow_stability,
        "current_ratio": current_ratio,
        "debt_to_equity_ratio": debt_to_equity_ratio,
        "profit_margin": profit_margin,
        "risk_tolerance": risk_tolerance
    }

# Function to select contract durations based on hedge requirements
def select_contract_durations(hedge_duration):
    return {
        "Short-Term (1-3 months)": [1, 2, 3],
        "Medium-Term (3-12 months)": [3, 6, 12],
        "Long-Term (12+ months)": [12, 18, 24]
    }[hedge_duration]

# Function to recommend contracts based on weighted correlations
def recommend_contracts(commodity_df, correlations, hedge_duration, financial_profile):
    contracts = []
    selected_durations = select_contract_durations(hedge_duration)

    for _, row in correlations.iterrows():
        commodity_data = commodity_df[commodity_df["Commodity"] == row["Commodity"]]
        commodity_data["Daily_Return"] = commodity_data["Commodity_Price"].pct_change()

        for duration in selected_durations:
            historical_data = commodity_data.tail(duration * 20)
            if historical_data.empty or len(historical_data) < 2:
                continue

            historical_return = historical_data["Daily_Return"].mean() * duration * 20
            historical_volatility = historical_data["Daily_Return"].std() * np.sqrt(252)
            expected_future_return, expected_future_volatility = forecast_returns_and_volatility(historical_data, duration)

            if expected_future_return > 0:
                weighted_return = expected_future_return * row["Correlation"]
                contracts.append({
                    "Duration (months)": duration,
                    "Expense_Category": row["Expense_Category"],
                    "Commodity": row["Commodity"],
                    "Correlation": row["Correlation"],
                    "Historical Average Return (%)": round(historical_return * 100, 2),
                    "Historical Volatility (%)": round(historical_volatility * 100, 2),
                    "Expected Future Return (%)": round(expected_future_return, 2),
                    "Expected Future Volatility (%)": round(expected_future_volatility, 2),
                    "Weighted Return": weighted_return
                })

    contracts_df = pd.DataFrame(contracts)
    return contracts_df.sort_values(by=["Weighted Return", "Expected Future Volatility (%)"], ascending=[False, True]).head(3)

# Streamlit UI setup
st.title("Hedging Strategy Dashboard")
st.sidebar.header("Upload Files")

# File uploader for QuickBooks and Commodity Data
quickbooks_file = st.sidebar.file_uploader("Upload QuickBooks Data (Excel)", type=["xlsx"])
commodity_file = st.sidebar.file_uploader("Upload Commodity Options Data (Excel)", type=["xlsx"])

if quickbooks_file and commodity_file:
    expense_df, cash_flow_df, balance_sheet_df, pnl_df = load_quickbooks_data(quickbooks_file)
    commodity_df = load_commodity_data(commodity_file)

    correlations = correlate_expenses_with_commodities(expense_df, commodity_df)
    financial_profile = assess_financial_profile(cash_flow_df, balance_sheet_df, pnl_df)
    volatility, high_season = calculate_volatility_and_seasonality(commodity_df)
    hedge_duration = recommend_hedge_duration(volatility, financial_profile, expense_df)

    recommended_contracts = recommend_contracts(commodity_df, correlations, hedge_duration, financial_profile)

    st.header("Top 3 Contract Recommendations")
    st.write(recommended_contracts)

    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    sns.barplot(x=recommended_contracts["Duration (months)"], y=recommended_contracts["Expected Future Return (%)"], ax=axs[0], color="lightblue", label="Expected Future Return")
    sns.lineplot(x=recommended_contracts["Duration (months)"], y=recommended_contracts["Expected Future Volatility (%)"], ax=axs[0], color="orange", marker="o", label="Expected Future Volatility")
    axs[0].set_title("Expected Future Return and Volatility by Duration")
    axs[0].legend()

    sns.barplot(x=recommended_contracts["Duration (months)"], y=recommended_contracts["Weighted Return"], ax=axs[1], color="green", label="Weighted Return")
    axs[1].set_title("Weighted Return by Duration")
    axs[1].legend()

    st.pyplot(fig)
else:
    st.info("Please upload both QuickBooks data and commodity options data files to proceed.")
