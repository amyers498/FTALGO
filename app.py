import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model  # Ensure 'arch' is installed for GARCH model
import seaborn as sns 

# Function to load financial data from QuickBooks
def load_quickbooks_data(file):
    # Load each sheet from QuickBooks Excel file into separate dataframes
    try:
        expense_df = pd.read_excel(file, sheet_name="Expense Report")
        cash_flow_df = pd.read_excel(file, sheet_name="Cash Flow Statement")
        balance_sheet_df = pd.read_excel(file, sheet_name="Balance Sheet")
        pnl_df = pd.read_excel(file, sheet_name="Profit & Loss Statement")
        return expense_df, cash_flow_df, balance_sheet_df, pnl_df
    except Exception as e:
        st.error("Error loading QuickBooks data. Please ensure the file has the correct sheets.")
        st.stop()

# Function to load commodity data from an Excel file
def load_commodity_data(file):
    try:
        return pd.read_excel(file)
    except Exception as e:
        st.error("Error loading Commodity data.")
        st.stop()

# Function to select primary expense category for hedging
def select_primary_cost(expense_df, manual_category=None):
    if manual_category:
        primary_cost_category = manual_category
        primary_cost = expense_df[expense_df["Expense_Category"] == manual_category]["Amount"].sum()
    else:
        expense_summary = expense_df.groupby("Expense_Category")["Amount"].sum().sort_values(ascending=False)
        primary_cost_category = expense_summary.idxmax()
        primary_cost = expense_summary.max()
    return primary_cost_category, primary_cost

# Automatically selects the best correlated commodity to expenses
def auto_select_commodity(expense_df, commodity_df):
    expense_monthly_totals = expense_df.groupby(expense_df['Date'].dt.to_period("M"))["Amount"].sum()
    correlations = {}
    for commodity in commodity_df["Commodity"].unique():
        commodity_prices = commodity_df[commodity_df["Commodity"] == commodity].set_index("Date")["Commodity_Price"]
        commodity_prices = commodity_prices.resample("M").mean()
        combined = pd.DataFrame({"Expense": expense_monthly_totals, "Commodity": commodity_prices}).dropna()
        correlation = combined["Expense"].corr(combined["Commodity"])
        correlations[commodity] = correlation
    best_commodity = max(correlations, key=correlations.get)
    return best_commodity, correlations[best_commodity]

# Function to calculate volatility and identify high season
def calculate_volatility_and_seasonality(commodity_df):
    commodity_df["Daily_Return"] = commodity_df["Commodity_Price"].pct_change()
    volatility = commodity_df["Daily_Return"].std() * np.sqrt(252)
    commodity_df["Month"] = pd.to_datetime(commodity_df["Date"]).dt.month
    monthly_avg = commodity_df.groupby("Month")["Commodity_Price"].mean()
    high_season = monthly_avg.idxmax() if monthly_avg.max() > 1.5 * monthly_avg.mean() else None
    return volatility * 100, high_season

# Function to predict returns and volatility using ARIMA and GARCH models
def forecast_returns_and_volatility(commodity_data, duration):
    model_arima = ARIMA(commodity_data['Commodity_Price'], order=(1, 1, 1))
    arima_fit = model_arima.fit()
    arima_forecast = arima_fit.forecast(steps=duration)
    expected_future_return = arima_forecast.mean() / commodity_data['Commodity_Price'].iloc[-1] - 1

    model_garch = arch_model(commodity_data['Daily_Return'].dropna(), vol='Garch', p=1, q=1)
    garch_fit = model_garch.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=duration)
    expected_future_volatility = np.sqrt(garch_forecast.variance.values[-1, :]).mean() * 100

    return expected_future_return * 100, expected_future_volatility

# Function to assess company's financial health and risk tolerance
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

# Function to select contract durations based on hedge requirements
def select_contract_durations(hedge_duration):
    contract_durations = {
        "Short-Term (1-3 months)": [1, 2, 3],
        "Medium-Term (3-12 months)": [3, 6, 12],
        "Long-Term (12+ months)": [12, 18, 24]
    }
    return contract_durations[hedge_duration]

# Function to recommend hedge duration based on volatility and risk tolerance
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

# Function to recommend contracts based on forecasts using ARIMA and GARCH models
def recommend_contracts_and_select_best(commodity_df, selected_commodity, hedge_duration, financial_profile):
    commodity_data = commodity_df[commodity_df['Commodity'] == selected_commodity]
    commodity_data["Daily_Return"] = commodity_data["Commodity_Price"].pct_change()
    selected_durations = select_contract_durations(hedge_duration)
    contracts = []

    for duration in selected_durations:
        historical_data = commodity_data.tail(duration * 20)
        if historical_data.empty or len(historical_data) < 2:
            continue

        historical_return = historical_data["Daily_Return"].mean() * duration * 20
        historical_volatility = historical_data["Daily_Return"].std() * np.sqrt(252)
        expected_future_return, expected_future_volatility = forecast_returns_and_volatility(historical_data, duration)

        contracts.append({
            "Duration (months)": duration,
            "Historical Average Return (%)": round(historical_return * 100, 2),
            "Historical Volatility (%)": round(historical_volatility * 100, 2),
            "Expected Future Return (%)": round(expected_future_return, 2),
            "Expected Future Volatility (%)": round(expected_future_volatility, 2)
        })

    sorted_contracts = sorted(
        contracts, 
        key=lambda x: x["Expected Future Return (%)"] / (x["Expected Future Volatility (%)"] + 1),
        reverse=True
    )
    return sorted_contracts[:3] if sorted_contracts else None

# Streamlit UI setup
st.title("Hedging Strategy Dashboard")
st.sidebar.header("Upload Files")

# File uploader for QuickBooks and Commodity Data
quickbooks_file = st.sidebar.file_uploader("Upload QuickBooks Data (Excel)", type=["xlsx"])
commodity_file = st.sidebar.file_uploader("Upload Commodity Options Data (Excel)", type=["xlsx"])

# If both files are uploaded, proceed with analysis
if quickbooks_file and commodity_file:
    expense_df, cash_flow_df, balance_sheet_df, pnl_df = load_quickbooks_data(quickbooks_file)
    commodity_df = load_commodity_data(commodity_file)

    # Sidebar option for selecting primary cost category
    st.sidebar.subheader("Primary Cost Category Selection")
    category_option = st.sidebar.selectbox("Choose primary cost selection method:", ["Auto-select", "Manual"])
    manual_category = None
    if category_option == "Manual":
        available_categories = expense_df["Expense_Category"].unique()
        manual_category         = st.sidebar.selectbox("Select a primary cost category:", available_categories)

    # Select primary cost category based on auto or manual choice
    primary_cost_category, primary_cost = select_primary_cost(expense_df, manual_category)

    # Sidebar option for selecting commodity for hedging
    st.sidebar.subheader("Commodity Selection for Hedging")
    commodity_option = st.sidebar.selectbox("Choose commodity selection method:", ["Auto-select", "Manual"])

    if commodity_option == "Manual":
        # Manual commodity selection
        selected_commodity = st.sidebar.selectbox("Select a commodity:", commodity_df["Commodity"].unique())
        correlation_value = "N/A (manual selection)"
    else:
        # Auto-select commodity with highest correlation
        selected_commodity, correlation_value = auto_select_commodity(expense_df, commodity_df)

    # Display analysis parameters
    st.header("Selected Analysis Parameters")
    st.write(f"**Primary Cost Category**: {primary_cost_category}")
    st.write(f"**Selected Commodity for Hedging**: {selected_commodity}")
    st.write(f"**Correlation with Expense**: {correlation_value:.2f}" if correlation_value != "N/A (manual selection)" else correlation_value)

    # Process data for the selected commodity
    selected_commodity_data = commodity_df[commodity_df["Commodity"] == selected_commodity]
    volatility, high_season = calculate_volatility_and_seasonality(selected_commodity_data)
    financial_profile = assess_financial_profile(cash_flow_df, balance_sheet_df, pnl_df)
    hedge_duration = recommend_hedge_duration(volatility, financial_profile, expense_df)
    top_contracts = recommend_contracts_and_select_best(commodity_df, selected_commodity, hedge_duration, financial_profile)

    # Display results of analysis
    st.write(f"**Commodity Volatility (%)**: {round(volatility, 2)}")
    st.write(f"**High Season**: {high_season}")
    st.write(f"**Recommended Hedge Duration**: {hedge_duration}")

    # Display top three contract recommendations
    if top_contracts:
        st.subheader("Top 3 Contract Recommendations")
        for idx, contract in enumerate(top_contracts, start=1):
            st.write(f"**Contract {idx}:**")
            st.write(contract)

        # Visualize historical and expected returns and volatilities
        contract_durations = [c["Duration (months)"] for c in top_contracts]
        historical_returns = [c["Historical Average Return (%)"] for c in top_contracts]
        historical_volatility = [c["Historical Volatility (%)"] for c in top_contracts]
        expected_returns = [c["Expected Future Return (%)"] for c in top_contracts]
        expected_volatility = [c["Expected Future Volatility (%)"] for c in top_contracts]

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Historical Return vs Volatility
        axs[0].bar(contract_durations, historical_returns, color='blue', label='Historical Return (%)')
        axs[0].plot(contract_durations, historical_volatility, color='red', marker='o', label='Historical Volatility (%)')
        axs[0].set_title("Historical Return and Volatility by Duration")
        axs[0].set_xlabel("Contract Duration (months)")
        axs[0].set_ylabel("Percentage (%)")
        axs[0].legend()

        # Expected Future Return vs Volatility
        axs[1].bar(contract_durations, expected_returns, color='blue', label='Expected Future Return (%)')
        axs[1].plot(contract_durations, expected_volatility, color='red', marker='o', label='Expected Future Volatility (%)')
        axs[1].set_title("Expected Future Return and Volatility by Duration")
        axs[1].set_xlabel("Contract Duration (months)")
        axs[1].set_ylabel("Percentage (%)")
        axs[1].legend()

        # Display the charts in Streamlit
        st.pyplot(fig)

    else:
        st.write("No viable contracts were found based on the given data.")
else:
    # Prompt user to upload both necessary files
    st.info("Please upload both QuickBooks data and commodity options data files to proceed.")

