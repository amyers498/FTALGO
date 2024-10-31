import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import seaborn as sns 

# Function to load financial data from QuickBooks
def load_quickbooks_data(file_path):
    expense_df = pd.read_excel(file_path, sheet_name="Expense Report")
    cash_flow_df = pd.read_excel(file_path, sheet_name="Cash Flow Statement")
    balance_sheet_df = pd.read_excel(file_path, sheet_name="Balance Sheet")
    pnl_df = pd.read_excel(file_path, sheet_name="Profit & Loss Statement")
    return expense_df, cash_flow_df, balance_sheet_df, pnl_df

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

# Function to load commodity data from an Excel file
def load_commodity_data(file_path):
    return pd.read_excel(file_path)

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

# Forecasting function using ARIMA and GARCH
def forecast_returns_and_volatility(commodity_data, duration_months):
    model_arima = ARIMA(commodity_data['Commodity_Price'], order=(1, 1, 1))
    arima_fit = model_arima.fit()
    arima_forecast = arima_fit.forecast(steps=duration_months)
    expected_future_returns = (arima_forecast / commodity_data['Commodity_Price'].iloc[-1] - 1) * 100

    model_garch = arch_model(commodity_data['Daily_Return'].dropna(), vol='Garch', p=1, q=1)
    garch_fit = model_garch.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=duration_months)
    expected_future_volatility = np.sqrt(garch_forecast.variance.values[-1]) * 100
    return expected_future_returns, expected_future_volatility

# Contract recommendation with ARIMA and GARCH
def recommend_contracts_and_select_best(commodity_df, selected_commodity, hedge_duration, financial_profile, primary_cost):
    commodity_data = commodity_df[commodity_df['Commodity'] == selected_commodity]
    commodity_data["Daily_Return"] = commodity_data["Commodity_Price"].pct_change()
    selected_durations = select_contract_durations(hedge_duration)
    contracts = []

    for duration in selected_durations:
        historical_data = commodity_data.tail(duration * 20)
        if historical_data.empty or len(historical_data) < 2:
            continue

        historical_returns = historical_data["Daily_Return"].mean() * duration * 20
        historical_volatility = historical_data["Daily_Return"].std() * np.sqrt(252)
        
        expected_future_returns, expected_future_volatility = forecast_returns_and_volatility(historical_data, duration)
        cumulative_future_return = expected_future_returns.sum()

        if cumulative_future_return > primary_cost:
            contracts.append({
                "Duration (months)": duration,
                "Historical Returns (%)": list(historical_returns * 100),
                "Historical Volatility (%)": list(historical_volatility * 100),
                "Expected Future Returns (%)": list(expected_future_returns),
                "Expected Future Volatility (%)": list(expected_future_volatility)
            })

    if contracts:
        best_contract = (min(contracts, key=lambda x: x["Expected Future Volatility (%)"]) if financial_profile["risk_tolerance"] == "High"
                         else max(contracts, key=lambda x: sum(x["Expected Future Returns (%)"]) / sum(x["Expected Future Volatility (%)"])))
        return best_contract, contracts
    else:
        return None, None

# Streamlit UI setup
st.title("Hedging Strategy Dashboard")
st.sidebar.header("Upload Files")

quickbooks_file = st.sidebar.file_uploader("Upload QuickBooks Data (Excel)", type=["xlsx"])
commodity_file = st.sidebar.file_uploader("Upload Commodity Options Data (Excel)", type=["xlsx"])

if quickbooks_file and commodity_file:
    expense_df, cash_flow_df, balance_sheet_df, pnl_df = load_quickbooks_data(quickbooks_file)
    commodity_df = load_commodity_data(commodity_file)

    st.sidebar.subheader("Primary Cost Category Selection")
    category_option = st.sidebar.selectbox("Choose primary cost selection method:", ["Auto-select", "Manual"])
    manual_category = None
    if category_option == "Manual":
        available_categories = expense_df["Expense_Category"].unique()
        manual_category = st.sidebar.selectbox("Select a primary cost category:", available_categories)

    primary_cost_category, primary_cost = select_primary_cost(expense_df, manual_category)

    st.sidebar.subheader("Commodity Selection for Hedging")
    commodity_option = st.sidebar.selectbox("Choose commodity selection method:", ["Auto-select", "Manual"])

    if commodity_option == "Manual":
        selected_commodity = st.sidebar.selectbox("Select a commodity:", commodity_df["Commodity"].unique())
        correlation_value = "N/A (manual selection)"
    else:
        selected_commodity, correlation_value = auto_select_commodity(expense_df, commodity_df)

    st.header("Selected Analysis Parameters")
    st.write(f"**Primary Cost Category**: {primary_cost_category}")
    st.write(f"**Selected Commodity for Hedging**: {selected_commodity}")
    st.write(f"**Primary Cost Value**: {primary_cost}")
    st.write(f"**Commodity Volatility (%)**: {round(volatility, 2)}")
    st.write(f"**High Season**: {high_season}")
    st.write(f"**Recommended Hedge Duration**: {hedge_duration}")

    selected_commodity_data = commodity_df[commodity_df["Commodity"] == selected_commodity]
    volatility, high_season = calculate_volatility_and_seasonality(selected_commodity_data)
    financial_profile = assess_financial_profile(cash_flow_df, balance_sheet_df, pnl_df)
    hedge_duration = recommend_hedge_duration(volatility, financial_profile, expense_df)
    best_contract, contracts = recommend_contracts_and_select_best(commodity_df, selected_commodity, hedge_duration, financial_profile, primary_cost)

    if best_contract:
        st.subheader("Best Contract Recommendation")
        st.write(best_contract)

        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        axs[0].plot(best_contract["Historical Returns (%)"], label='Monthly Historical Return (%)', color='blue')
        axs[0].plot(best_contract["Historical Volatility (%)"], label='Monthly Historical Volatility (%)', color='red')
        axs[0].set_title("Monthly Historical Return and Volatility")
        axs[0].legend()
        axs[0].set_xlabel("Months")
        axs[0].set_ylabel("Percentage (%)")

        axs[1].plot(best_contract["Expected Future Returns (%)"], label='Monthly Expected Future Return (%)', color='green')
        axs[1].plot(best_contract["Expected Future Volatility (%)"], label='Monthly Expected Future Volatility (%)', color='purple')
        axs[1].set_title("Monthly Expected Future Return and Volatility")
        axs[1].legend()
        axs[1].set_xlabel("Months")
        axs[1].set_ylabel("Percentage (%)")

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No viable contracts were found based on the given data.")
else:
    st.info("Please upload both QuickBooks data and commodity options data files to proceed.")
