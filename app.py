import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Load the saved models and everything they need
lin_model = joblib.load("house_linear_model.pkl")
log_model = joblib.load("house_logistic_model.pkl")
scaler = joblib.load("house_scaler.pkl")
feature_columns = joblib.load("house_feature_columns.pkl")
medians = joblib.load("house_medians.pkl")
threshold = joblib.load("house_threshold.pkl")
neighborhoods = joblib.load("house_neighborhoods.pkl")
with open("house_metrics.json") as f:
    metrics = json.load(f)

st.title("House Price Prediction")
tab1, tab2, tab3 = st.tabs(["Predict", "Model Performance", "Dataset"])

# ---------------- Tab 1: Prediction ----------------
with tab1:
    st.write("Enter the main details of the house. All other features are set to the "
             "typical (median) house from the training data, so this is an approximate estimate.")
    c1, c2 = st.columns(2)

    with c1:
        overall_qual = st.slider("Overall quality (1 = poor, 10 = excellent)", 1, 10, 6)
        gr_liv_area = st.number_input("Living area above ground (sq ft)", 300, 6000, 1500)
        total_bsmt = st.number_input("Total basement area (sq ft)", 0, 6000, 1000)
        garage_cars = st.selectbox("Garage capacity (cars)", [0, 1, 2, 3, 4], index=2)
        lot_area = st.number_input("Lot area (sq ft)", 1000, 100000, 9500)

    with c2:
        year_built = st.number_input("Year built", 1870, 2026, 1975)
        full_bath = st.selectbox("Full bathrooms", [0, 1, 2, 3, 4], index=2)
        bedrooms = st.selectbox("Bedrooms above ground", [0, 1, 2, 3, 4, 5, 6], index=3)
        fireplaces = st.selectbox("Fireplaces", [0, 1, 2, 3], index=1)
        neighborhood = st.selectbox("Neighborhood", neighborhoods)

    if st.button("Predict price"):
        # Start from the typical house, then overwrite the values the user entered
        row = medians.copy()
        row["OverallQual"] = overall_qual
        row["GrLivArea"] = gr_liv_area
        row["TotalBsmtSF"] = total_bsmt
        row["GarageCars"] = garage_cars
        row["LotArea"] = lot_area
        row["YearBuilt"] = year_built
        row["FullBath"] = full_bath
        row["BedroomAbvGr"] = bedrooms
        row["Fireplaces"] = fireplaces

        # Keep the related year columns sensible for a newer house
        row["YearRemodAdd"] = max(year_built, medians["YearRemodAdd"])
        row["GarageYrBlt"] = max(year_built, medians["GarageYrBlt"])

        # Neighborhood became 0/1 columns: reset them all, then set the chosen one
        for col in feature_columns:
            if col.startswith("Neighborhood_"):
                row[col] = 0
        nb_col = f"Neighborhood_{neighborhood}"
        if nb_col in row.index:   # the first neighborhood was dropped, so all 0 means that one
            row[nb_col] = 1

        # Same column order as training, then scale
        input_df = pd.DataFrame([row])[feature_columns]
        input_scaled = scaler.transform(input_df)

        # Linear regression predicts log(price): convert back to dollars
        price = np.expm1(lin_model.predict(input_scaled))[0]
        prob_expensive = log_model.predict_proba(input_scaled)[0][1]
        mae = metrics["linear"]["mae"]

        st.success(f"Estimated price: ${price:,.0f}")
        st.write(f"Typical error of this model: about ±${mae:,.0f}")
        st.write(f"Probability the house is above the median price "
                 f"(${threshold:,.0f}): **{prob_expensive:.0%}**")
        st.caption("Estimate based on the Ames, Iowa housing dataset (prices in US dollars). "
                   "Student ML project, not a real valuation.")

# ---------------- Tab 2: Model performance ----------------
with tab2:
    st.subheader("Linear regression (predicts the price)")
    lin = metrics["linear"]
    a, b = st.columns(2)
    a.metric("R² (price scale)", f"{lin['r2_price']:.3f}")
    b.metric("R² (log scale)", f"{lin['r2_log']:.3f}")
    c, d = st.columns(2)
    c.metric("MAE", f"${lin['mae']:,.0f}")
    d.metric("RMSE", f"${lin['rmse']:,.0f}")

    st.subheader("Logistic regression (expensive or not)")
    logi = metrics["logistic"]
    st.write(f"**Accuracy:** {logi['accuracy']:.3f}")

    report = dict(logi["report"])
    report.pop("accuracy", None)
    st.write("**Classification report** (class 0 = not expensive, class 1 = expensive)")
    st.dataframe(pd.DataFrame(report).T.round(2))

    st.write("**Confusion matrix**")
    cm = pd.DataFrame(
        logi["confusion_matrix"],
        index=["Actual: Not expensive", "Actual: Expensive"],
        columns=["Predicted: Not expensive", "Predicted: Expensive"])
    st.dataframe(cm)

    st.subheader("Comparison on the expensive / not-expensive question")
    st.dataframe(pd.DataFrame({
        "Model": ["Logistic regression", "Linear regression (cut at median price)"],
        "Accuracy": [round(logi["accuracy"], 3),
                     round(metrics["linear_as_classifier_accuracy"], 3)],
    }), hide_index=True)

# ---------------- Tab 3: Dataset ----------------
with tab3:
    st.subheader("Ames housing dataset")
    data = pd.read_csv("house_prices.csv")
    st.write(f"{data.shape[0]} rows and {data.shape[1]} columns (raw data, before cleaning)")

    st.write("**First 20 rows**")
    st.dataframe(data.head(20))

    st.write("**Summary statistics**")
    st.dataframe(data.describe())

    st.write("**SalePrice distribution** (right-skewed, the reason we used a log transform)")
    counts = pd.cut(data["SalePrice"], bins=20).value_counts().sort_index()
    counts.index = counts.index.astype(str)
    st.bar_chart(counts)

    st.write("**Median price by neighborhood**")
    st.bar_chart(data.groupby("Neighborhood")["SalePrice"].median().sort_values())