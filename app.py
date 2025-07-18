import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import xgboost as xgb
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Maintain State
if "df" not in st.session_state:
    st.session_state.df = None
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# UI
st.title("Personal Expense Tracker")
st.subheader("Just upload a CSV file with these required columns and press the button!")

image = Image.open("format.png")
st.image(image)

required_columns = ["DATE", "CATEGORY", "DESCRIPTION", "AMOUNT", "PAYMENT_METHOD", 
                    "MERCHANT", "IS_NECESSARY"]

uploaded_file = st.file_uploader(label="Upload CSV file here", type="csv")

# When Analyze Button is pressed
if st.button("Analyze"):
    if uploaded_file is None:
        st.warning("Please upload a CSV file first.")
        st.stop()

    df = pd.read_csv(uploaded_file)

    # Check column names (MUST BE IN ORDER!!!)
    if list(df.columns.str.upper()) != required_columns:
        st.warning("Please follow the columns (and its order) from the sample image.")
        st.stop()

    # Check missing values
    if df.isnull().values.any():
        st.warning("File contains missing values.")
        st.stop()

    # Convert DATE column and use ISO Format for consistency
    df.columns = df.columns.str.upper()
    if not df["DATE"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").all():
        st.warning("DATE column must be in YYYY-MM-DD format (e.g., 2023-06-29).")
        st.stop()

    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d", errors="coerce")
    if df["DATE"].isna().any():
        st.warning("DATE column contains invalid date values.")
        st.stop()
    
    # Check if amount is all float
    df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors='coerce')
    if df["AMOUNT"].isna().any():
        st.warning("AMOUNT column contains invalid values.")
        st.stop()

    # Check if yes or no for converting to binary values for ml
    df["IS_NECESSARY"] = df["IS_NECESSARY"].astype(str).str.lower().map({"yes": 1, "no": 0})
    if df["IS_NECESSARY"].isna().any():
        st.warning("IS_NECESSARY must contain only 'yes' or 'no'.")
        st.stop()

    # Convert CATEGORY and PAYMENT_METHOD to categorical
    df["CATEGORY"] = df["CATEGORY"].astype("category")
    df["PAYMENT_METHOD"] = df["PAYMENT_METHOD"].astype("category")

    # Compilation of needed columns for feature engineering in ML
    df["MONTH_YEAR"] = df["DATE"].dt.to_period("M").astype(str)
    df["DAY"] = df["DATE"].dt.day_name()
    df["WEEKDAY"] = df["DATE"].dt.weekday
    df["WEEKEND"] = df["WEEKDAY"].isin([5,6]).astype(int)
    df["WEEK_NUM"] = df["DATE"].dt.isocalendar().week
    df["YEAR"] = df["DATE"].dt.year
    df["YEAR_MONTH"] = df["DATE"].dt.to_period("M")

    weekly_spend = df.groupby(["YEAR", "WEEK_NUM"])["AMOUNT"].sum().reset_index()
    df = df.merge(weekly_spend, on=["YEAR", "WEEK_NUM"], suffixes=('', '_WEEKLY'))

    avg_weekly_spend = weekly_spend["AMOUNT"].mean()
    df["OVERSPENDING"] = (df["AMOUNT_WEEKLY"] > avg_weekly_spend).astype(int)

    st.session_state.df = df
    st.session_state.analyzed = True
    st.success("File processed successfully!")

# Display
if st.session_state.analyzed and st.session_state.df is not None:
    df = st.session_state.df

    # Preview
    st.subheader("Preview of Uploaded CSV File")
    preview_df = df[required_columns].copy()
    preview_df["DATE"] = preview_df["DATE"].dt.strftime("%Y-%m-%d")
    preview_df["IS_NECESSARY"] = preview_df["IS_NECESSARY"].replace({1: "Yes", 0: "No"})
    st.write(preview_df.head())

    # ===========================================================================================
    # Spending Trend Chart (Line Plot)
    st.subheader("Spending Trend Chart")
    total_spend = df.groupby("MONTH_YEAR", as_index=False)["AMOUNT"].sum()
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#111", "figure.facecolor": "#111"})
    sns.lineplot(x="MONTH_YEAR", y="AMOUNT", data=total_spend, marker='o')
    plt.xlabel("MONTH/YEAR", color="white")
    plt.ylabel("AMOUNT PER MONTH/YEAR", color='white')
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    st.pyplot(plt)
    plt.clf()

    # Min & Max
    min_row = total_spend.loc[total_spend["AMOUNT"].idxmin()]
    formatted_min = f"{min_row['AMOUNT']:.2f} ({min_row['MONTH_YEAR']})"

    max_row = total_spend.loc[total_spend["AMOUNT"].idxmax()]
    formatted_max = f"{max_row['AMOUNT']:.2f} ({max_row['MONTH_YEAR']})"

    st.text(f"Minimum Month/Year Spent: {formatted_min}")
    st.text(f"Maximum Month/Year Spent: {formatted_max}")

    # Spending for Each Day of Week (Bar Plot)
    st.subheader("Total Spending for Each Day of the Week")
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    spend_per_day = df.groupby("DAY", as_index=False)["AMOUNT"].sum()
    sns.barplot(x="DAY", y="AMOUNT", data=spend_per_day, order=day_order)
    plt.xlabel("DAY OF WEEK", color="white")
    plt.ylabel("AMOUNT", color="white")
    plt.xticks(rotation=60, color="white")
    plt.yticks(color="white")
    st.pyplot(plt)
    plt.clf()

    # Min & Max 2
    min_row2 = spend_per_day.loc[spend_per_day["AMOUNT"].idxmin()]
    formatted_min2 = f"{min_row2['AMOUNT']:.2f} ({min_row2['DAY']})"

    max_row2 = spend_per_day.loc[spend_per_day["AMOUNT"].idxmax()]
    formatted_max2 = f"{max_row2['AMOUNT']:.2f} ({max_row2['DAY']})"

    st.text(f"Day with Lowest Total Amount Spent: {formatted_min2}")
    st.text(f"Day with Highest Total Amount Spent: {formatted_max2}")

    # Yes vs No (Pie Chart)
    st.subheader("Yes vs No in 'IS_NECESSARY' Column")
    counts = df["IS_NECESSARY"].value_counts().sort_index()
    plt.figure(facecolor="#111")
    plt.pie(counts, labels=["No", "Yes"], autopct='%1.2f%%', textprops={'color':'white'})
    st.pyplot(plt)
    plt.clf()

    # Total Spent
    all_time_spent = df["AMOUNT"].sum()
    st.text(f"Total Amount Spent (All Time): {all_time_spent:,.2f}")

    # ===========================================================================================
    # Expense Prediction with Model Comparison and Best Model Insights

    st.header("Expense Prediction Section")

    # Preparing data (Train test split)
    ml_df = df[["CATEGORY", "PAYMENT_METHOD", "IS_NECESSARY", "DAY", "WEEKDAY", "WEEKEND", "AMOUNT"]].copy()
    ml_df = pd.get_dummies(ml_df, columns=["CATEGORY", "PAYMENT_METHOD", "DAY"], drop_first=True)

    X = ml_df.drop("AMOUNT", axis=1)
    y = ml_df["AMOUNT"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Defining models
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "KNN Regressor": make_pipeline(StandardScaler(), KNeighborsRegressor()),
        "XGBoost Regressor": xgb.XGBRegressor(random_state=42)
    }

    # Train and evaluate models
    mse_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_results[name] = (mse, model, y_pred)

    # Display MSEs (might comment out)
    #st.markdown("### Model Evaluation (MSE Scores)")
    #for name, (mse, _, _) in mse_results.items():
        #st.text(f"{name}: MSE = {mse:.2f}")

    # Pick the best model
    best_model_name = min(mse_results, key=lambda name: mse_results[name][0])
    best_mse, best_model, best_y_pred = mse_results[best_model_name]

    #st.success(f"Best performing model: **{best_model_name}** (MSE: {best_mse:.2f})") # might remove MSE value
    st.success(f"Best performing model: **{best_model_name}**")
    #st.markdown("This model had the lowest average error when predicting your expenses.") # might remove this as well

    # Show Feature Importance
    if best_model_name in ["Random Forest Regressor", "XGBoost Regressor"]:
        st.subheader("Feature Importance")
        st.text(f"{best_model_name} Regression MSE: {best_mse:.2f}")

        fig, ax = plt.subplots()
        if best_model_name == "Random Forest Regressor":
            importances = best_model.feature_importances_
            features = X.columns
            sns.barplot(x=importances, y=features, ax=ax)
            ax.set_title("Feature Importances")
        else: 
            xgb.plot_importance(best_model, ax=ax, height=0.5, max_num_features=10)

            for text in ax.texts:
                text.set_visible(False)
            ax.title.set_color('white')
        
        # Making it white to contrast from dark mode
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        fig.patch.set_facecolor('#111'); ax.set_facecolor('#111')
        st.pyplot(fig)
        plt.clf()

        st.info("""
        This chart shows which features influenced the prediction the most.
        Use this insight to understand what drives your expenses. 🎯
        """)

    # Forecasting 
    st.subheader("Next Week Spend Forecast")
    last_week = df["WEEK_NUM"].max()
    last_year = df["YEAR"].max()
    recent_weeks = df[(df["YEAR"]==last_year) & (df["WEEK_NUM"] > last_week - 3)]
    next_week_forecast = (recent_weeks["AMOUNT"].sum() / 3)
    st.text(f"Forecasted spend for next week (week {last_week+1}): {next_week_forecast:.2f}")

    # Aggregating weekly spending
    df_weekly = df.groupby(["YEAR", "WEEK_NUM"]).agg({"AMOUNT": "sum"}).reset_index()

    # Create lag features and target for regression
    df_weekly["LAG_1"] = df_weekly["AMOUNT"].shift(1)
    df_weekly["LAG_2"] = df_weekly["AMOUNT"].shift(2)
    df_weekly["LAG_3"] = df_weekly["AMOUNT"].shift(3)
    df_weekly["TARGET"] = df_weekly["AMOUNT"].shift(-1)  # Next week's spend

    df_weekly.dropna(inplace=True)

    # Train another model using RFR
    X = df_weekly[["LAG_1", "LAG_2", "LAG_3"]]
    y = df_weekly["TARGET"]
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X, y)

    # Use most recent data to predict next week
    latest = df_weekly.tail(1)[["LAG_1", "LAG_2", "LAG_3"]].values

    # Simulate model uncertainty using estimators
    predictions = np.array([est.predict(latest)[0] for est in model.estimators_])
    mean_prediction = predictions.mean()
    prob_high_spend = (predictions >= next_week_forecast).mean()

    st.text(f"Model's average prediction for next week: {mean_prediction:.2f}")
    st.info(f"Probability that spend exceeds {next_week_forecast:.2f} is {prob_high_spend:.2%}")

    # ===========================================================================================
    # Anomaly Detection (via Z-score)
    st.subheader("Unusual Transaction Detection")
    amount_mean = df["AMOUNT"].mean()
    amount_std = df["AMOUNT"].std()
    df["AMOUNT_ZSCORE"] = (df["AMOUNT"] - amount_mean) / amount_std
    abnormal_txns = df[df["AMOUNT_ZSCORE"].abs() > 2]
    st.write(f"Number of questionable transactions: {len(abnormal_txns)}")
    st.dataframe(abnormal_txns[["DATE", "CATEGORY", "DESCRIPTION", "AMOUNT", "AMOUNT_ZSCORE"]].round(2))

    # Personalized Suggestions
    st.subheader("Personalized Suggestions")
    top_categories = df.groupby("CATEGORY")["AMOUNT"].mean().sort_values(ascending=False).head(3)
    st.write("Top 3 categories by AVERAGE spend per transaction:")
    st.write(top_categories.round(2))

    weekend_avg = df[df["WEEKEND"]==1]["AMOUNT"].mean()
    weekday_avg = df[df["WEEKEND"]==0]["AMOUNT"].mean()
    if weekend_avg > weekday_avg:
        st.warning(f"You spend more on weekends (Avg: {weekend_avg:.2f}) than weekdays (Avg: {weekday_avg:.2f}).")
    else:
        st.success(f"Your weekday spending (Avg: {weekday_avg:.2f}) is higher which is expected.")

    monthly_totals = df.groupby("YEAR_MONTH")["AMOUNT"].sum().sort_index()
    if len(monthly_totals) >= 2:
        last_val, current_val = monthly_totals.iloc[-2], monthly_totals.iloc[-1]
        change = ((current_val - last_val) / last_val) * 100
        if change > 0:
            st.warning(f"You spent {change:.2f}% more this month.")
        else:
            st.success(f"You've reduced your spending by {abs(change):.2f}% from last month!")

    # Show Actual vs Predicted
    st.subheader("Predictions vs Actual (Sample)")
    sample_df = X_test.copy()
    sample_df["Actual"] = y_test
    sample_df["Predicted"] = np.round(best_y_pred,2)

    st.write(sample_df[["Actual", "Predicted"]].head(10))
    st.info("""
    📊 The table above shows a comparison between your **actual expenses** and what the model predicted.

    If the prediction is **close**, it means your spending is consistent.  
    If it's **far off**, it may mean you spent more or less than usual — something worth checking or reflecting on.
    """)

    # Personalized Automatic Budget Recommendations
    st.subheader("Auto Budget Recommendations")
    enable_recommendation = st.toggle("Generate Personalized Category Budgets")

    if enable_recommendation:
        latest_date = df["DATE"].max()
        recent_df = df[df["DATE"] >= latest_date - pd.DateOffset(months=3)]
        if recent_df.empty or recent_df["CATEGORY"].nunique() < 2:
            st.warning("Not enough recent data for recommendations.")
        else:
            non_essential = recent_df[recent_df["IS_NECESSARY"] == 0]
            avg_spend = non_essential.groupby("CATEGORY")["AMOUNT"].mean().sort_values(ascending=False)
            budget = (avg_spend * 0.85).round(2)
            st.write("Suggested Monthly Caps (15% Reduction):")
            st.dataframe(budget.rename("Suggested Budget"))