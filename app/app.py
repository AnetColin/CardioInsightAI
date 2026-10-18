
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.inspection import permutation_importance


# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="CardioInsight AI",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ CardioInsight AI")

st.subheader(
    "Cardiovascular Risk Factors: "
    "Data Analytics & Machine Learning"
)

st.markdown(
    """
    Explore cardiovascular health data, discover patterns in
    risk factors, and compare machine learning models.

    This project combines **Exploratory Data Analysis (EDA)**,
    **Logistic Regression**, and **Random Forest** to study
    patterns in the dataset.
    """
)

st.info(
    "Educational research prototype only. "
    "Model outputs are not medical diagnoses or validated "
    "individual risk predictions."
)

# ---------------- LOAD DATA AND TRAIN MODELS ----------------
@st.cache_resource
def load_and_train():
    project_dir = Path(__file__).resolve().parent.parent
    csv_path = project_dir / "data" / "P5-Cardio Vascular Risk Prediction.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. "
            "Place the CSV inside the project's data folder."
        )

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.dropna(subset=["TenYearCHD"])

    X = df.drop(columns=["TenYearCHD", "id"], errors="ignore")
    y = df["TenYearCHD"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    numeric_cols = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, pred, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, prob)
        }

        predictions[name] = {
            "pred": pred,
            "prob": prob,
            "matrix": confusion_matrix(y_test, pred)
        }

    importance_result = permutation_importance(
        models["Logistic Regression"],
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=3,
        random_state=42,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": importance_result.importances_mean,
        "Std": importance_result.importances_std
    }).sort_values("Importance", ascending=False)

    return (
    df, X, X_test, y_test, results,
    predictions, importance_df,models
)


# ---------------- RUN ANALYSIS ----------------
try:
    with st.spinner("Loading data and training models..."):
        df,X, X_test, y_test, results, predictions, importance_df,models = (
            load_and_train()
        )
except Exception as exc:
    st.error(f"Could not start the analysis: {exc}")
    st.stop()


# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Data Analytics",
        "Model Lab",
        "Feature Insights",
        "Prediction Lab",
        "Test Known Records"
    ]
)

# ---------------- DATA ANALYTICS ----------------
if section == "Data Analytics":
    st.header("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Records", f"{len(df):,}")
    col2.metric("Features", len(df.columns) - 1)
    col3.metric(
        "Missing Cells",
        f"{int(df.isnull().sum().sum()):,}"
    )

    st.write("Preview of the dataset")
    st.dataframe(df.head(10), use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Ten-Year CHD Outcome Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="TenYearCHD", ax=ax)
        ax.set_xlabel("Recorded outcome (0 or 1)")
        ax.set_ylabel("Number of records")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x="age", bins=20, kde=True, ax=ax)
        ax.set_xlabel("Age")
        ax.set_ylabel("Number of records")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Recorded Measurements by Outcome")

    available_columns = [
        c for c in ["totChol", "sysBP", "diaBP", "BMI", "glucose"]
        if c in df.columns
    ]

    selected_column = st.selectbox(
        "Choose a measurement",
        available_columns
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=df,
        x="TenYearCHD",
        y=selected_column,
        ax=ax
    )
    ax.set_xlabel("Recorded outcome (0 or 1)")
    ax.set_title(f"{selected_column} by recorded outcome")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "These charts describe associations in this dataset; "
        "they do not establish causation."
    )


# ---------------- MODEL LAB ----------------
elif section == "Model Lab":
    st.header("🤖 Model Evaluation")

    metrics_df = pd.DataFrame(results).T
    st.write("Evaluation on the held-out test set")
    st.dataframe(
        metrics_df.style.format("{:.3f}"),
        use_container_width=True
    )

    metric = st.selectbox(
        "Compare models using",
        ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    )
    st.subheader("Understanding the Results")
    
    st.markdown("""
    - **Accuracy:** Overall proportion of correct predictions.
    - **Precision:** Of the cases predicted positive, how many
      were actually positive?
    - **Recall:** Of the actual positive cases, how many did
      the model identify?
    - **F1-score:** Balance between precision and recall.
    - **ROC-AUC:** Measures how well the model ranks positive
      cases above negative cases across classification thresholds.
    """)
    
    st.info(
        "In our earlier experiment, Logistic Regression achieved "
        "higher recall, F1-score, and ROC-AUC than Random Forest. "
        "This is a research comparison, not evidence of clinical "
        "effectiveness."
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics_df[metric].plot(kind="bar", ax=ax)
    ax.set_title(f"Model Comparison — {metric}")
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close(fig)

    selected_model = st.selectbox(
        "View confusion matrix",
        list(predictions.keys())
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        predictions[selected_model]["matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax
    )
    ax.set_title(selected_model)
    st.pyplot(fig)
    plt.close(fig)

    st.warning(
        "Model scores are preliminary and depend on this dataset and "
        "split. They do not establish clinical suitability."
    )


# ---------------- FEATURE INSIGHTS ----------------
# ---------------- FEATURE INSIGHTS 

# ---------------- FEATURE INSIGHTS ----------------
elif section == "Feature Insights":
    st.header("🔎 Feature Insights")

    st.write(
        "Permutation importance estimates how much the model's ROC-AUC "
        "changes when one feature is shuffled."
    )

    top_n = st.slider(
        "Number of features to display",
        min_value=5,
        max_value=min(12, len(importance_df)),
        value=min(10, len(importance_df))
    )

    display_df = importance_df.head(top_n).sort_values(
        "Importance", ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        display_df["Feature"],
        display_df["Importance"],
        xerr=display_df["Std"]
    )
    ax.set_xlabel("Decrease in ROC-AUC after shuffling")
    ax.set_title("Permutation Importance — Logistic Regression")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("What Do These Features Tell Us?")

    st.markdown("""
    Permutation importance estimates how much a model's
    performance changes when the values of a feature are shuffled.

    - **Higher importance:** Shuffling the feature reduces model
      performance more in this evaluation.
    - **Lower importance:** Shuffling the feature has less effect
      on the model's performance.
    - **Interpret carefully:** Importance does not prove that a
      feature causes cardiovascular disease.
    """)

    st.info(
        "These results describe patterns learned by the model "
        "from this dataset. They do not establish medical causes "
        "or predict an individual's actual health outcome."
    )

    st.dataframe(
        importance_df.round(4),
        use_container_width=True
    )

    st.caption(
        "Importance measures model dependence, not causation. "
        "Correlated features can share or obscure importance."
    )

        
# ---------------- PREDICTION LAB ----------------
elif section == "Prediction Lab":
    st.header("🧪 Prediction Lab")

    st.write(
        "Enter values from the dataset to compare the outputs "
        "of the two trained models."
    )

    st.warning(
        "Educational demonstration only. These model outputs "
        "are not medical diagnoses or validated personal risk estimates."
    )

    with st.form("prediction_form"):
        user_input = {}
        input_columns = st.columns(2)

        for i, feature in enumerate(X.columns):
            series = X[feature]

            with input_columns[i % 2]:
                if pd.api.types.is_numeric_dtype(series):
                    clean = series.dropna().astype(float)

                    low = float(clean.min())
                    high = float(clean.max())
                    default = float(clean.median())

                    if low == high:
                        low -= 1
                        high += 1

                    is_integer = np.all(
                        np.isclose(clean % 1, 0)
                    )
                    step = 1.0 if is_integer else 0.1

                    user_input[feature] = st.number_input(
                        feature,
                        min_value=low,
                        max_value=high,
                        value=min(max(default, low), high),
                        step=step
                    )
                else:
                    options = series.dropna().unique().tolist()

                    if options:
                        user_input[feature] = st.selectbox(
                            feature,
                            options=options,
                            format_func=str
                        )

        submitted = st.form_submit_button(
            "Compare Model Predictions",
            type="primary"
        )

    if submitted:
        input_df = pd.DataFrame(
            [user_input],
            columns=X.columns
        )

        st.subheader("Prediction Results")

        result_columns = st.columns(2)

        for i, (name, model) in enumerate(models.items()):
            predicted = int(model.predict(input_df)[0])
            score = float(model.predict_proba(input_df)[0, 1])

            with result_columns[i]:
                st.markdown(f"**{name}**")

                st.metric(
                    "Predicted dataset label",
                    predicted
                )

                st.metric(
                    "Model score for label 1",
                    f"{score:.1%}"
                )

        st.caption(
            "Label 0 and label 1 represent the dataset's recorded "
            "target classes. The model score is not a validated "
            "estimate of an individual's actual medical risk."
        )


# ---------------- TEST KNOWN RECORDS ----------------
elif section == "Test Known Records":
    st.header("✅ Test Against Known Outcomes")

    st.write(
        "Evaluate the models on records from the held-out test set. "
        "Their actual target labels are known, so we can measure "
        "how many predictions are correct."
    )

    st.info(
        "These records were held out from model training. "
        "This is a demonstration of dataset performance, not "
        "clinical validation."
    )

    number_of_records = st.slider(
        "Number of test records to evaluate",
        min_value=1,
        max_value=len(X_test),
        value=min(20, len(X_test))
    )

    if st.button("Evaluate Test Records", type="primary"):
        X_subset = X_test.sample(
            n=number_of_records,
            random_state=42
        )
        y_subset = y_test.loc[X_subset.index]

        comparison = pd.DataFrame({
            "Actual label": y_subset
        })

        accuracy_columns = st.columns(2)

        for i, (name, model) in enumerate(models.items()):
            predicted = model.predict(X_subset)

            correct = (predicted == y_subset.to_numpy())
            accuracy = accuracy_score(
                y_subset,
                predicted
            )

            with accuracy_columns[i]:
                st.metric(
                    f"{name} accuracy on selected records",
                    f"{accuracy:.1%}"
                )
                st.write(
                    f"Correct predictions: {int(correct.sum())} "
                    f"out of {number_of_records}"
                )

            comparison[f"{name} prediction"] = predicted
            comparison[f"{name} correct?"] = np.where(
                correct, "Yes", "No"
            )

        st.subheader("Actual vs Predicted Labels")
        st.dataframe(
            comparison.reset_index(drop=True),
            use_container_width=True
        )

        st.caption(
            "Accuracy on a small selected sample can vary. "
            "Use the full held-out test-set results in Model Lab "
            "for the main model comparison."
        )

        st.divider()
        st.caption("CardioInsight AI | Student ML & Data Analytics Project")

