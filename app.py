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

# Determine hedge contract duration based on expense volatility
def determine_contract_duration(expense_df, category):
    monthly_expenses = expense_df[expense_df["Expense_Category"] == category].groupby(expense_df["Date"].dt.to_period("M"))["Amount"].sum()
    volatility = monthly_expenses.std() / monthly_expenses.mean()
    
    if volatility > 0.2:
        return 3  # Short-term for high volatility
    elif 0.1 <= volatility <= 0.2:
        return 6  # Medium-term for moderate volatility
    else:
        return 12  # Long-term for low volatility

# Correlate all expenses with all commodities
def correlate_expenses_with_commodities(expense_df, commodity_df):
    expense_monthly_totals = expense_df.groupby([expense_df['Date'].dt.to_period("M"), 'Expense_Category'])["Amount"].sum().unstack(fill_value=0)
    correlations = {}

    for commodity in commodity_df["Commodity"].unique():
        commodity_prices = commodity_df[commodity_df["Commodity"] == commodity].set_index("Date")["Commodity_Price"].resample("M").mean()
        for category in expense_monthly_totals.columns:
            combined = pd.DataFrame({"Expense": expense_monthly_totals[category], "Commodity": commodity_prices}).dropna()
            correlation = combined["Expense"].corr(combined["Commodity"])
            correlations[(commodity, category)] = correlation

    return sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

# Correlate a specific expense category with all commodities
def correlate_specific_category(expense_df, commodity_df, category):
    expense_monthly_totals = expense_df.groupby([expense_df['Date'].dt.to_period("M"), 'Expense_Category'])["Amount"].sum().unstack(fill_value=0)
    correlations = {}

    if category not in expense_monthly_totals.columns:
        st.warning(f"No data found for the selected category: {category}")
        return []

    for commodity in commodity_df["Commodity"].unique():
        commodity_prices = commodity_df[commodity_df["Commodity"] == commodity].set_index("Date")["Commodity_Price"].resample("M").mean()
        combined = pd.DataFrame({"Expense": expense_monthly_totals[category], "Commodity": commodity_prices}).dropna()
        correlation = combined["Expense"].corr(combined["Commodity"])
        correlations[(commodity, category)] = correlation

    return sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

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

# Recommend contracts based on best correlations, using expense-based contract length
def recommend_best_hedges(commodity_df, expense_df, best_correlations):
    recommendations = []

    for (commodity, category), correlation in best_correlations:
        commodity_data = commodity_df[commodity_df['Commodity'] == commodity]
        commodity_data["Daily_Return"] = commodity_data["Commodity_Price"].pct_change()

        # Set contract duration based on volatility
        duration = determine_contract_duration(expense_df, category)

        monthly_returns, volatility = forecast_returns_and_volatility(commodity_data, duration)
        avg_monthly_expense = expense_df[expense_df["Expense_Category"] == category]["Amount"].mean()

        estimated_savings = avg_monthly_expense * (monthly_returns.mean() / 100) * duration
        if estimated_savings > 0:
            recommendations.append({
                "Commodity": commodity,
                "Category": category,
                "Correlation": round(correlation, 2),
                "Expected Monthly Return (%)": monthly_returns.mean(),
                "Expected Volatility (%)": volatility.mean(),
                "Potential Savings": round(estimated_savings, 2),
                "Duration (Months)": duration,
                "Explanation": f"This hedge was selected for {category} due to a {round(correlation * 100, 1)}% correlation with {commodity}. A {duration}-month contract was suggested based on expense volatility."
            })

    return recommendations[:3]

# Streamlit Setup
st.title("Enhanced Hedging Strategy Dashboard")
quickbooks_file = st.sidebar.file_uploader("Upload QuickBooks Data (Excel)", type=["xlsx"])
commodity_file = st.sidebar.file_uploader("Upload Commodity Options Data (Excel)", type=["xlsx"])

if quickbooks_file and commodity_file:
    expense_df, cash_flow_df, balance_sheet_df, pnl_df = load_quickbooks_data(quickbooks_file)
    commodity_df = load_commodity_data(commodity_file)

    # Monthly and average expenses
    st.header("Company's Expense Overview")
    last_12_months = expense_df[expense_df['Date'] >= (expense_df['Date'].max() - pd.DateOffset(months=12))]
    avg_monthly_expense = last_12_months.groupby('Expense_Category')["Amount"].mean().sum()

    st.write(f"**Average Monthly Expenses:** ${avg_monthly_expense:.2f}")
    st.write("### Monthly Expenses by Category in the Past 12 Months")
    monthly_expenses = last_12_months.groupby([last_12_months['Date'].dt.to_period("M"), 'Expense_Category'])["Amount"].sum().unstack()

    # Display bar chart for monthly expenses by category
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_expenses.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title("Monthly Expenses by Category (Last 12 Months)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Expense Amount ($)")
    st.pyplot(fig)

    # Sidebar for selection mode
    st.sidebar.subheader("Hedging Options")
    selection_mode = st.sidebar.selectbox("Selection Mode", ["Auto-select Best", "Specific Category"])

    if selection_mode == "Specific Category":
        selected_category = st.sidebar.selectbox("Choose Expense Category", expense_df["Expense_Category"].unique())
        specific_correlations = correlate_specific_category(expense_df, commodity_df, selected_category)
        best_recommendations = recommend_best_hedges(commodity_df, expense_df, specific_correlations)

        # Display recommendations
        st.subheader(f"Hedge Recommendations for {selected_category}")
        for idx, rec in enumerate(best_recommendations, start=1):
            st.write(f"### Recommendation {idx}")
            st.write(f"**Commodity:** {rec['Commodity']}")
            st.write(f"**Correlation with Expense:** {rec['Correlation']}")
            st.write(f"**Expected Monthly Return (%):** {rec['Expected Monthly Return (%)']:.2f}")
            st.write(f"**Expected Volatility (%):** {rec['Expected Volatility (%)']:.2f}")
            st.write(f"**Potential Savings:** ${rec['Potential Savings']:.2f}")
            st.write(f"**Duration (Months):** {rec['Duration (Months)']}")
            st.write(f"**Explanation:** {rec['Explanation']}")

            # Plot Monthly Returns
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=range(1, len(rec['Monthly Returns']) + 1), y=rec['Monthly Returns'], marker="o", ax=ax)
            ax.set_title(f"Monthly Expected Returns for {rec['Commodity']} in {rec['Category']}")
            ax.set_xlabel("Month")
            ax.set_ylabel("Expected Return (%)")
            st.pyplot(fig)

    else:
        # Auto-select the best correlations
        best_correlations = correlate_expenses_with_commodities(expense_df, commodity_df)
        best_recommendations = recommend_best_hedges(commodity_df, expense_df, best_correlations)

        # Display top recommendations
        st.subheader("Top Hedge Recommendations")
        for idx, rec in enumerate(best_recommendations, start=1):
            st.write(f"### Recommendation {idx}")
            st.write(f"**Commodity:** {rec['Commodity']}")
            st.write(f"**Expense Category:** {rec['Category']}")
            st.write(f"**Correlation with Expense:** {rec['Correlation']}")
            st.write(f"**Expected Monthly Return (%):** {rec['Expected Monthly Return (%)']:.2f}")
            st.write(f"**Expected Volatility (%):** {rec['Expected Volatility (%)']:.2f}")
            st.write(f"**Potential Savings:** ${rec['Potential Savings']:.2f}")
            st.write(f"**Duration (Months):** {rec['Duration (Months)']}")
            st.write(f"**Explanation:** {rec['Explanation']}")

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

        if not best_recommendations:
            st.write("No viable contracts were found. Please review the data or adjust criteria.")

else:
    st.info("Please upload both QuickBooks data and commodity options data files to proceed.")

