import pickle
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Price Predictor", layout="wide")

@st.cache_resource
def load_model():
    with open("car_price_model.pickle", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("eda_train.csv")

bundle = load_model()
data = load_data()

model = bundle["model"]
encoder = bundle["encoder"]
numeric_cols = bundle["numeric_cols"]
categorical_cols = bundle["categorical_cols"]
feature_columns = bundle["feature_columns"]

def preprocess(df):
    df = df.copy()

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(
            df[col].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
            errors="coerce"
        )
        df[col] = df[col].fillna(bundle["numeric_medians"][col])

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = np.nan

        if col == "seats":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(bundle["numeric_medians"][col])
        else:
            df[col] = df[col].fillna(bundle["category_fallbacks"][col])

    encoded = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    X = pd.concat([df[numeric_cols], encoded], axis=1)
    return X.reindex(columns=feature_columns, fill_value=0)


st.title("Car Price Prediction Dashboard")

tab1, tab2, tab3 = st.tabs([
    "EDA Dashboard",
    "Price Prediction",
    "Model Coefficients"
])

with tab1:
    st.header("Exploratory Data Analysis")

    st.subheader("Car Price Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data["selling_price"], bins=40, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Selling Price vs Year")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="year", y="selling_price", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        data.select_dtypes(include="number").corr(),
        annot=True, cmap="coolwarm", fmt=".2f", ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.header("Predict Car Prices")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.dataframe(input_data.head())

        if st.button("Predict Prices"):
            try:
                X = preprocess(input_data)
                predictions = model.predict(X)

                results = input_data.copy()
                results["predicted_price"] = predictions

                st.success("Predictions completed!")
                st.dataframe(results)

                st.download_button(
                    "Download Predictions",
                    results.to_csv(index=False).encode("utf-8"),
                    "predictions.csv",
                    "text/csv"
                )
            except Exception as e:
                st.error(f"Prediction error: {e}")

with tab3:
    st.header("Model Coefficients")

    coefficients = pd.Series(
        model.coef_,
        index=feature_columns
    ).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    coefficients.plot(kind="barh", ax=ax)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Ridge Regression Feature Weights")

    st.pyplot(fig)
    plt.close(fig)
