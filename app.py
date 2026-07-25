import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Telco Customer Churn — Agentic Dashboard")
st.markdown("Upload your dataset and the pipeline runs automatically.")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")

# Allow upload OR auto-load from data/ folder
_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "data", "Telco_customer_churn.xlsx")
uploaded_file = st.sidebar.file_uploader("Upload Telco_customer_churn.xlsx", type=["xlsx"])
if uploaded_file is None and os.path.exists(_DEFAULT_PATH):
    st.sidebar.info("Using bundled dataset from `data/` folder.")
    uploaded_file = _DEFAULT_PATH  # pass path string; handled below

model_choice = st.sidebar.selectbox(
    "Model for Churn Prediction",
    ["Random Forest", "Gradient Boosting", "Logistic Regression"]
)

n_clusters = st.sidebar.slider("Number of Segments (k)", 2, 8, 3)

run_btn = st.sidebar.button("🚀 Run Full Pipeline", use_container_width=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
DROP_COLS = [
    'CustomerID','Count','Country','State','City',
    'Zip Code','Lat Long','Latitude','Longitude',
    'Churn Score','CLTV','Churn Reason','Churn Label'
]

@st.cache_data
def load_and_preprocess(file):
    # Accept either an uploaded file object or a file path string
    df = pd.read_excel(file)
    raw = df.copy()

    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'].fillna(df['Total Charges'].median(), inplace=True)

    existing_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=existing_drop, inplace=True)

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=['Churn Value'])
    Y = df['Churn Value']
    return raw, X, Y

def train_model(X_train, Y_train, choice):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10,
            random_state=42, class_weight='balanced'
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        )
    }
    m = models[choice]
    m.fit(X_train, Y_train)
    return m

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
if uploaded_file and run_btn:

    raw_df, X, Y = load_and_preprocess(uploaded_file)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # ── TABS ──────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 EDA", "🏆 Model", "📊 Evaluation", "🗂️ Segments", "💡 Recommendations"
    ])

    # ══════════════════════════════════════════
    # TAB 1 — EDA
    # ══════════════════════════════════════════
    with tab1:
        st.header("🔍 Exploratory Data Analysis")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", raw_df.shape[0])
        col2.metric("Total Features", raw_df.shape[1])
        churn_pct = round(raw_df['Churn Label'].value_counts(normalize=True).get('Yes', 0) * 100, 1)
        col3.metric("Churn Rate", f"{churn_pct}%")
        col4.metric("Non-Churn Rate", f"{100 - churn_pct}%")

        st.subheader("Data Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Churn Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))
            raw_df['Churn Label'].value_counts().plot(
                kind='bar', ax=ax, color=['steelblue', 'salmon']
            )
            ax.set_title("Churn Label Counts")
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

        with col_b:
            st.subheader("Tenure Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(raw_df['Tenure Months'], bins=30, kde=True, ax=ax, color='steelblue')
            ax.set_title("Tenure Months")
            plt.tight_layout()
            st.pyplot(fig)

        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Monthly Charges Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(raw_df['Monthly Charges'], bins=30, kde=True, ax=ax, color='green')
            plt.tight_layout()
            st.pyplot(fig)

        with col_d:
            st.subheader("Correlation Heatmap")
            df_temp = raw_df.copy()
            df_temp['Total Charges'] = pd.to_numeric(df_temp['Total Charges'], errors='coerce')
            num_cols = ['Tenure Months','Monthly Charges','Total Charges',
                        'Churn Value','Churn Score','CLTV']
            existing = [c for c in num_cols if c in df_temp.columns]
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(df_temp[existing].corr(), annot=True, fmt='.2f',
                        cmap='coolwarm', ax=ax)
            plt.tight_layout()
            st.pyplot(fig)

        st.subheader("Missing Values")
        missing = raw_df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            st.success("✅ No missing values found!")
        else:
            st.warning(f"⚠️ {len(missing)} columns have missing values")
            st.bar_chart(missing)

    # ══════════════════════════════════════════
    # TAB 2 — MODEL
    # ══════════════════════════════════════════
    with tab2:
        st.header("🏆 Model Training & Comparison")

        with st.spinner("Training all 3 models..."):
            all_models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=300, max_depth=10,
                    random_state=42, class_weight='balanced'
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=200, max_depth=5, random_state=42
                ),
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=42, class_weight='balanced'
                )
            }
            results = []
            trained = {}
            for name, m in all_models.items():
                m.fit(X_train, Y_train)
                yp = m.predict(X_test)
                trained[name] = m
                results.append({
                    'Model': name,
                    'Accuracy':  round(accuracy_score(Y_test, yp), 4),
                    'Recall':    round(recall_score(Y_test, yp), 4),
                    'Precision': round(precision_score(Y_test, yp), 4),
                    'F1 Score':  round(f1_score(Y_test, yp), 4),
                    'ROC-AUC':   round(roc_auc_score(Y_test, m.predict_proba(X_test)[:, 1]), 4)
                })

        result_df = pd.DataFrame(results).sort_values('Recall', ascending=False)
        st.subheader("Model Comparison Table")
        st.dataframe(result_df.set_index('Model'), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        metrics = ['Accuracy','Recall','Precision','F1 Score','ROC-AUC']
        result_df.set_index('Model')[metrics].T.plot(kind='bar', ax=ax, colormap='Set2')
        ax.set_title("All Models — All Metrics")
        ax.set_ylabel("Score")
        ax.legend(loc='lower right')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

        best_name = result_df.iloc[0]['Model']
        st.success(f"🏆 Best model by Recall: **{best_name}**")

        # Use user-selected model going forward
        best_model = trained[model_choice]
        st.info(f"📌 Using **{model_choice}** (your selection) for predictions below.")

    # ══════════════════════════════════════════
    # TAB 3 — EVALUATION
    # ══════════════════════════════════════════
    with tab3:
        st.header("📊 Model Evaluation")

        model = trained[model_choice]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{accuracy_score(Y_test, y_pred):.2%}")
        col2.metric("Recall",    f"{recall_score(Y_test, y_pred):.2%}")
        col3.metric("Precision", f"{precision_score(Y_test, y_pred):.2%}")
        col4.metric("F1 Score",  f"{f1_score(Y_test, y_pred):.2%}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(Y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Churn','Churn'],
                        yticklabels=['No Churn','Churn'])
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
            ax.set_title(f"Confusion Matrix — {model_choice}")
            plt.tight_layout()
            st.pyplot(fig)

        with col_b:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(Y_test, y_prob)
            auc = roc_auc_score(Y_test, y_prob)
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color='steelblue', label=f'AUC = {auc:.3f}')
            ax.plot([0,1],[0,1],'k--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        if hasattr(model, 'feature_importances_'):
            st.subheader("Top 15 Feature Importances")
            imp = pd.Series(model.feature_importances_, index=X.columns)
            imp = imp.sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10, 4))
            imp.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title("Feature Importances")
            plt.tight_layout()
            st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(Y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 4 — SEGMENTS
    # ══════════════════════════════════════════
    with tab4:
        st.header("🗂️ Customer Segmentation")

        model = trained[model_choice]
        df_temp = raw_df.copy()
        df_temp['Total Charges'] = pd.to_numeric(df_temp['Total Charges'], errors='coerce')
        df_temp['Total Charges'].fillna(df_temp['Total Charges'].median(), inplace=True)

        churn_prob_all = model.predict_proba(X)[:, 1]
        seg_df = pd.DataFrame({
            'Tenure Months':    df_temp['Tenure Months'].values,
            'Monthly Charges':  df_temp['Monthly Charges'].values,
            'Total Charges':    df_temp['Total Charges'].values,
            'Churn Probability': churn_prob_all
        })

        scaler = StandardScaler()
        scaled = scaler.fit_transform(seg_df)

        with st.spinner("Running KMeans + Silhouette..."):
            wcss, sil = [], []
            for k in range(2, 10):
                km = KMeans(n_clusters=k, random_state=42)
                lbl = km.fit_predict(scaled)
                wcss.append(km.inertia_)
                sil.append(silhouette_score(scaled, lbl))

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(range(2, 10), wcss, marker='o', color='steelblue')
            ax.set_title("Elbow Method")
            ax.set_xlabel("k"); ax.set_ylabel("WCSS")
            plt.tight_layout(); st.pyplot(fig)
        with col_b:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(range(2, 10), sil, marker='o', color='green')
            ax.axvline(n_clusters, color='red', linestyle='--', label=f'k={n_clusters}')
            ax.set_title("Silhouette Score")
            ax.set_xlabel("k"); ax.legend()
            plt.tight_layout(); st.pyplot(fig)

        km_final = KMeans(n_clusters=n_clusters, random_state=42)
        seg_df['Cluster'] = km_final.fit_predict(scaled)

        # Auto-name by churn probability
        cp_rank = seg_df.groupby('Cluster')['Churn Probability'].mean().sort_values()
        names = {}
        for i, c in enumerate(cp_rank.index):
            if i == 0:
                names[c] = 'Loyal Low-Risk'
            elif i == len(cp_rank) - 1:
                names[c] = 'High-Risk'
            else:
                names[c] = f'Medium-Risk {i}'
        seg_df['Segment'] = seg_df['Cluster'].map(names)

        st.subheader("Cluster Summary")
        st.dataframe(seg_df.groupby('Segment').mean().round(2), use_container_width=True)

        col_c, col_d = st.columns(2)
        with col_c:
            fig, ax = plt.subplots(figsize=(5, 4))
            for seg, grp in seg_df.groupby('Segment'):
                ax.scatter(grp['Monthly Charges'], grp['Churn Probability'],
                           label=seg, alpha=0.4, s=8)
            ax.set_xlabel("Monthly Charges")
            ax.set_ylabel("Churn Probability")
            ax.set_title("Monthly Charges vs Churn Probability")
            ax.legend(fontsize=7)
            plt.tight_layout(); st.pyplot(fig)

        with col_d:
            seg_counts = seg_df['Segment'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            seg_counts.plot(kind='bar', ax=ax, color=['steelblue','salmon','green','orange'])
            ax.set_title("Customers per Segment")
            ax.set_ylabel("Count")
            plt.xticks(rotation=20)
            plt.tight_layout(); st.pyplot(fig)

        # Store seg_df in session for Tab 5
        st.session_state['seg_df'] = seg_df

    # ══════════════════════════════════════════
    # TAB 5 — RECOMMENDATIONS
    # ══════════════════════════════════════════
    with tab5:
        st.header("💡 Business Recommendations")

        if 'seg_df' not in st.session_state:
            st.warning("Run segmentation first (Tab 4).")
        else:
            seg_df = st.session_state['seg_df']
            seg_df['Revenue at Risk'] = (
                seg_df['Monthly Charges'] * seg_df['Churn Probability']
            )

            profile = seg_df.groupby('Segment').agg(
                Customers       = ('Segment','count'),
                Avg_Churn_Prob  = ('Churn Probability','mean'),
                Avg_Monthly     = ('Monthly Charges','mean'),
                Revenue_at_Risk = ('Revenue at Risk','sum')
            ).round(2)

            st.subheader("Segment Profiles")
            st.dataframe(profile, use_container_width=True)

            st.subheader("💰 Monthly Revenue at Risk")
            fig, ax = plt.subplots(figsize=(7, 3))
            profile['Revenue_at_Risk'].sort_values().plot(
                kind='barh', ax=ax, color=['green','orange','red']
            )
            ax.set_title("Revenue at Risk per Segment ($)")
            plt.tight_layout(); st.pyplot(fig)

            st.subheader("📋 Recommended Actions")
            for seg in profile.index:
                cp = profile.loc[seg, 'Avg_Churn_Prob']
                n  = int(profile.loc[seg, 'Customers'])
                rev = profile.loc[seg, 'Revenue_at_Risk']

                if 'High' in seg:
                    color = '🔴'
                    actions = [
                        "Immediate personalised outreach (phone/email within 48h)",
                        "Offer loyalty discount or contract upgrade",
                        "Assign dedicated account manager"
                    ]
                elif 'Medium' in seg:
                    color = '🟡'
                    actions = [
                        "Send satisfaction survey",
                        "Offer add-on services (streaming, security)",
                        "Enrol in loyalty / referral programme"
                    ]
                else:
                    color = '🟢'
                    actions = [
                        "Upsell to premium plans",
                        "Referral incentives",
                        "Reward with anniversary offers"
                    ]

                with st.expander(f"{color} {seg} — {n} customers | Churn Prob: {cp:.0%} | Revenue at Risk: ${rev:,.0f}"):
                    for a in actions:
                        st.write(f"→ {a}")

            st.subheader("📌 Executive Summary")
            profile['Action'] = profile.index.map(
                lambda s: 'IMMEDIATE' if 'High' in s else ('MONITOR' if 'Medium' in s else 'UPSELL')
            )
            st.dataframe(profile, use_container_width=True)

else:
    st.info("👈 Upload your Excel file and click **Run Full Pipeline** to start.")
    st.markdown("""
    ### What this dashboard does:
    | Tab | Description |
    |-----|-------------|
    | 🔍 EDA | Auto explores your data — distributions, correlations, missing values |
    | 🏆 Model | Trains 3 models and compares them automatically |
    | 📊 Evaluation | Confusion matrix, ROC curve, feature importances |
    | 🗂️ Segments | KMeans clustering with auto optimal k selection |
    | 💡 Recommendations | Revenue at risk + actionable business strategies |
    """)

