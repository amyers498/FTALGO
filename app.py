import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model  # Ensure 'arch' is installed for GARCH model
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

# Correlate multiple expenses with commodities
def correlate_expenses_with_commodities(expense_df, commodity_df):
    expense_monthly_totals = expense_df.groupby([expense_df['Date'].dt.to_period("M"), 'Expense_Category'])["Amount"].sum().unstack(fill_value=0)
    correlations = {}

    for commodity in commodity_df["Commodity"].unique():
        commodity_prices = commodity_df[commodity_df["Commodity"] == commodity].set_index("Date")["Commodity_Price"]
        commodity_prices = commodity_prices.resample("M").mean()
        for category in expense_monthly_totals.columns:
            combined = pd.DataFrame({"Expense": expense_monthly_totals[category], "Commodity": commodity_prices}).dropna()
            correlation = combined["Expense"].corr(combined["Commodity"])
            correlations[(commodity, category)] = correlation

    # Sort by correlation value and return top commodities with associated categories
    best_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    return best_correlations[:3]  # Top 3 best correlations

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

# Recommend contracts based on best expense-commodity correlations
def recommend_best_hedges(commodity_df, expense_df, best_correlations):
    recommendations = []

    for (commodity, category), correlation in best_correlations:
        commodity_data = commodity_df[commodity_df['Commodity'] == commodity]
        commodity_data["Daily_Return"] = commodity_data["Commodity_Price"].pct_change()
        duration = 12  # Example duration, could be adjusted

        monthly_returns, volatility = forecast_returns_and_volatility(commodity_data, duration)
        avg_monthly_expense = expense_df[expense_df["Expense_Category"] == category]["Amount"].mean()

        # Calculate potential savings over duration
        estimated_savings = avg_monthly_expense * (monthly_returns.mean() / 100) * duration
        recommendations.append({
            "Commodity": commodity,
            "Category": category,
            "Correlation": round(correlation, 2),
            "Expected Monthly Return (%)": monthly_returns.mean(),
            "Expected Volatility (%)": volatility.mean(),
            "Potential Savings": round(estimated_savings, 2),
            "Monthly Returns": monthly_returns.tolist()
        })

    return recommendations

# Streamlit Setup
st.title("Enhanced Hedging Strategy Dashboard")
quickbooks_file = st.sidebar.file_uploader("Upload QuickBooks Data (Excel)", type=["xlsx"])
commodity_file = st.sidebar.file_uploader("Upload Commodity Options Data (Excel)", type=["xlsx"])

# If both files are uploaded, proceed with analysis
if quickbooks_file and commodity_file:
    expense_df, cash_flow_df, balance_sheet_df, pnl_df = load_quickbooks_data(quickbooks_file)
    commodity_df = load_commodity_data(commodity_file)

    # Sidebar options for selecting expense and commodity or auto-selection
    st.sidebar.subheader("Hedging Options")
    selection_mode = st.sidebar.selectbox("Selection Mode", ["Auto-select Best", "Manual Selection"])

    if selection_mode == "Manual Selection":
        # Manual selection of category and commodity
        manual_category = st.sidebar.selectbox("Select Expense Category", expense_df["Expense_Category"].unique())
        manual_commodity = st.sidebar.selectbox("Select Commodity", commodity_df["Commodity"].unique())

        selected_commodity_data = commodity_df[commodity_df["Commodity"] == manual_commodity]
        selected_commodity_data["Daily_Return"] = selected_commodity_data["Commodity_Price"].pct_change()
        
        # Forecast returns and volatility for manual selection
        duration = 12  # Example duration for manual choice
        monthly_returns, volatility = forecast_returns_and_volatility(selected_commodity_data, duration)
        avg_monthly_expense = expense_df[expense_df["Expense_Category"] == manual_category]["Amount"].mean()
        estimated_savings = avg_monthly_expense * (monthly_returns.mean() / 100) * duration

        # Display manual selection analysis
        st.subheader(f"Manual Hedge Analysis for {manual_category} with {manual_commodity}")
        st.write(f"**Expected Monthly Return (%):** {monthly_returns.mean():.2f}")
        st.write(f"**Expected Volatility (%):** {volatility.mean():.2f}")
        st.write(f"**Potential Savings over Duration:** ${estimated_savings:.2f}")

        # Plot Expected Monthly Returns for manual selection
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=range(1, duration + 1), y=monthly_returns, marker="o", ax=ax)
        ax.set_title(f"Monthly Expected Returns for {manual_commodity} in {manual_category}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Expected Return (%)")
        st.pyplot(fig)

        # Plot Potential Savings
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=[manual_category], y=[estimated_savings], ax=ax, color="green")
        ax.set_title(f"Estimated Total Savings for {manual_category} Expense")
        ax.set_xlabel("Expense Category")
        ax.set_ylabel("Savings ($)")
        st.pyplot(fig)

    else:
        # Auto-select the best correlations and recommend hedges
        best_correlations = correlate_expenses_with_commodities(expense_df, commodity_df)
        best_recommendations = recommend_best_hedges(commodity_df, expense_df, best_correlations)

        # Display top contract recommendations
        if best_recommendations:
            st.subheader("Top Hedge Recommendations")
            sns.set_theme(style="whitegrid")

            for idx, rec in enumerate(best_recommendations, start=1):
                st.write(f"### Recommendation {idx}")
                st.write(f"**Commodity:** {rec['Commodity']}")
                st.write(f"**Expense Category:** {rec['Category']}")
                st.write(f"**Correlation with Expense:** {rec['Correlation']}")
                st.write(f"**Expected Monthly Return (%):** {rec['Expected Monthly Return (%)']:.2f}")
                st.write(f"**Expected Volatility (%):** {rec['Expected Volatility (%)']:.2f}")
                st.write(f"**Estimated Savings over Duration:** ${rec['Potential Savings']:.2f}")

                # Plot Monthly Returns
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(x=range(1, len(rec['Monthly Returns']) + 1), y=rec['Monthly Returns'], marker="o", ax=ax)
                ax.set_title(f"Monthly Expected Returns for {rec['Commodity']} in {rec['Category']}")
                ax.set_xlabel("Month")
                ax.set_ylabel("Expected Return (%)")
                st.pyplot(fig)

                # Plot Potential Savings
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=[rec['Category']], y=[rec['Potential Savings']], ax=ax, color="green")
                ax.set_title(f"Estimated Total Savings for {rec['Category']} Expense")
                ax.set_xlabel("Expense Category")
                ax.set_ylabel("Savings ($)")
                st.pyplot(fig)

        else:
            st.write("No viable contracts were found based on the given data.")
else:
    st.info("Please upload both QuickBooks data and commodity options data files to proceed.")
