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

# Correlate all expenses with all commodities
# Correlate all expenses with all commodities
# Debugging alignment and date overlap
def correlate_expenses_with_commodities(expense_df, commodity_df):
    # Convert dates to datetime if necessary
    expense_df['Date'] = pd.to_datetime(expense_df['Date'])
    commodity_df['Date'] = pd.to_datetime(commodity_df['Date'])

    # Calculate monthly totals for expenses
    expense_monthly_totals = expense_df.groupby([expense_df['Date'].dt.to_period("M"), 'Expense_Category'])["Amount"].sum().unstack(fill_value=0)
    expense_monthly_totals.index = expense_monthly_totals.index.to_timestamp()

    correlations = {}

    for commodity in commodity_df["Commodity"].unique():
        # Resample commodity prices to monthly averages
        commodity_prices = commodity_df[commodity_df["Commodity"] == commodity].set_index("Date")["Commodity_Price"].resample("M").mean()

        # Check date range for debugging
        st.write(f"Commodity '{commodity}' date range:", commodity_prices.index.min(), "to", commodity_prices.index.max())

        for category in expense_monthly_totals.columns:
            # Combine and align data, drop NaNs
            combined = pd.DataFrame({"Expense": expense_monthly_totals[category], "Commodity": commodity_prices}).dropna()

            # Debug output for combined data
            st.write(f"Combined data for {commodity} and {category}:")
            st.write(combined.head())

            if combined.empty:
                st.write(f"No overlap in data for {commodity} and {category}")
                correlations[(commodity, category)] = None
            else:
                correlation = combined["Expense"].corr(combined["Commodity"])
                correlations[(commodity, category)] = correlation
                st.write(f"Correlation between {commodity} and {category}: {correlation}")

    best_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]) if x[1] is not None else 0, reverse=True)
    return best_correlations



# Correlate a specific expense category with all commodities
def correlate_specific_category(expense_df, commodity_df, category):
    expense_monthly_totals = expense_df.groupby([expense_df['Date'].dt.to_period("M"), 'Expense_Category'])["Amount"].sum().unstack()
    expense_monthly_totals.index = expense_monthly_totals.index.to_timestamp()
    
    correlations = {}

    if category not in expense_monthly_totals.columns:
        st.warning(f"No data found for the selected category: {category}")
        return []

    for commodity in commodity_df["Commodity"].unique():
        # Resample and align commodity prices
        commodity_prices = commodity_df[commodity_df["Commodity"] == commodity].set_index("Date")["Commodity_Price"].resample("M").mean().to_period("M")
        
        # Combine and align data without filling missing values with 0
        combined = pd.DataFrame({"Expense": expense_monthly_totals[category].astype(float), "Commodity": commodity_prices.astype(float)})
        combined.dropna(inplace=True)  # Drop rows with NaN values

        # Calculate correlation
        correlation = combined["Expense"].corr(combined["Commodity"])
        correlations[(commodity, category)] = correlation

    # Sort and return top correlations
    best_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    return best_correlations[:3]

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

# Recommend contracts based on best correlations, ensuring positive savings
def recommend_best_hedges(commodity_df, expense_df, best_correlations):
    recommendations = []

    for (commodity, category), correlation in best_correlations:
        commodity_data = commodity_df[commodity_df['Commodity'] == commodity]
        commodity_data["Daily_Return"] = commodity_data["Commodity_Price"].pct_change()
        duration = 12  # Example duration; can be adjusted based on QuickBooks data analysis

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
                "Monthly Returns": monthly_returns.tolist() if not monthly_returns.empty else None,
                "Explanation": f"This hedge was selected for {category} due to a {round(correlation * 100, 1)}% correlation with {commodity}, offering stable returns and helping to control price fluctuations."
            })

    if not recommendations:
        st.write("No positive savings recommendations found.")
    return recommendations[:3]

# Streamlit Setup (the rest of the setup remains unchanged)

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
            st.write(f"**Explanation:** {rec['Explanation']}")

            # Check for Monthly Returns data before plotting
            if rec['Monthly Returns']:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(x=range(1, len(rec['Monthly Returns']) + 1), y=rec['Monthly Returns'], marker="o", ax=ax)
                ax.set_title(f"Monthly Expected Returns for {rec['Commodity']} in {rec['Category']}")
                ax.set_xlabel("Month")
                ax.set_ylabel("Expected Return (%)")
                st.pyplot(fig)
            else:
                st.write("No Monthly Returns data available for plotting.")

    else:
        # Auto-select the best correlations and recommend hedges
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
            st.write(f"**Explanation:** {rec['Explanation']}")

            # Check for Monthly Returns data before plotting
            if rec['Monthly Returns']:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(x=range(1, len(rec['Monthly Returns']) + 1), y=rec['Monthly Returns'], marker="o", ax=ax)
                ax.set_title(f"Monthly Expected Returns for {rec['Commodity']} in {rec['Category']}")
                ax.set_xlabel("Month")
                ax.set_ylabel("Expected Return (%)")
                st.pyplot(fig)
            else:
                st.write("No Monthly Returns data available for plotting.")

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

