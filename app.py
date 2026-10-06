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
    page_title="Churn Prediction & Segmentation Dashboard",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Universal Telco Churn & Segmentation Intelligence Dashboard")
st.markdown("Upload any customer dataset (Excel or CSV). The machine learning pipeline runs automatically.")

# ─────────────────────────────────────────────
# SIDEBAR & DATA LOADING
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")

possible_paths = [
    os.path.join(os.path.dirname(__file__), "churn_app", "data", "Telco_customer_churn.xlsx"),
    os.path.join(os.path.dirname(__file__), "data", "Telco_customer_churn.xlsx"),
    os.path.join(os.path.dirname(__file__), "Telco_customer_churn.xlsx"),
    "Telco_customer_churn.xlsx",
]
_DEFAULT_PATH = None
for p in possible_paths:
    if os.path.exists(p):
        _DEFAULT_PATH = p
        break

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (.xlsx, .xls, .csv)",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is None and _DEFAULT_PATH and os.path.exists(_DEFAULT_PATH):
    st.sidebar.info("Using bundled default dataset (`Telco_customer_churn.xlsx`).")
    uploaded_file = _DEFAULT_PATH

model_choice = st.sidebar.selectbox(
    "Model for Churn Prediction",
    ["Random Forest", "Gradient Boosting", "Logistic Regression"]
)

n_clusters = st.sidebar.slider("Number of Segments (k)", 2, 8, 3)

# ─────────────────────────────────────────────
# ROBUST HELPER FUNCTIONS
# ─────────────────────────────────────────────

def _clean_str(s):
    return "".join(c for c in str(s).lower() if c.isalnum())

@st.cache_data
def load_raw_dataframe(file):
    if isinstance(file, str):
        if file.lower().endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    else:
        filename = file.name.lower()
        if filename.endswith('.csv'):
            try:
                return pd.read_csv(file)
            except Exception:
                file.seek(0)
                return pd.read_csv(file, encoding='latin1')
        else:
            try:
                return pd.read_excel(file)
            except Exception:
                file.seek(0)
                return pd.read_csv(file)

def auto_detect_columns(df):
    clean_cols = {_clean_str(c): c for c in df.columns}
    
    # Target
    target_col = None
    priority_targets = ['churnvalue', 'churnlabel', 'churn', 'target', 'exited', 'churned', 'is_churn', 'churn_flag', 'label', 'class', 'response']
    for key in priority_targets:
        if key in clean_cols:
            target_col = clean_cols[key]
            break
    if not target_col:
        for c in df.columns:
            cl = c.lower()
            if 'churn' in cl or 'target' in cl or 'exited' in cl:
                target_col = c
                break
    if not target_col:
        binary_cols = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
        target_col = binary_cols[-1] if binary_cols else df.columns[-1]

    # Tenure
    tenure_col = None
    for k in ['tenuremonths', 'tenure_months', 'tenure', 'months', 'period', 'accountlength', 'duration']:
        if k in clean_cols:
            tenure_col = clean_cols[k]
            break
    if not tenure_col:
        for c in df.columns:
            if 'tenure' in c.lower() or 'month' in c.lower():
                tenure_col = c
                break

    # Monthly Charges
    monthly_col = None
    for k in ['monthlycharges', 'monthly_charges', 'monthlycharge', 'monthly_charge', 'monthlyamount', 'monthlypay', 'monthlyrate']:
        if k in clean_cols:
            monthly_col = clean_cols[k]
            break
    if not monthly_col:
        for c in df.columns:
            if 'monthly' in c.lower() or ('charge' in c.lower() and c != tenure_col):
                monthly_col = c
                break

    # Total Charges
    total_col = None
    for k in ['totalcharges', 'total_charges', 'totalcharge', 'total_charge', 'totalamount', 'totalpay', 'totalrevenue', 'accumulatedcharges']:
        if k in clean_cols:
            total_col = clean_cols[k]
            break
    if not total_col:
        for c in df.columns:
            if 'total' in c.lower() and c not in [tenure_col, monthly_col]:
                total_col = c
                break

    return target_col, tenure_col, monthly_col, total_col

def preprocess_dataset(raw_df, target_col, tenure_col, monthly_col, total_col):
    df = raw_df.copy()

    # Target (Y)
    y_raw = df[target_col]
    if pd.api.types.is_numeric_dtype(y_raw) and set(y_raw.dropna().unique()).issubset({0, 1}):
        y_binary = y_raw.astype(int)
    else:
        y_str = y_raw.astype(str).str.strip().str.lower()
        pos_words = {'yes', 'true', '1', 'churned', 'y', '1.0', 'positive', 'pos', 'exited'}
        if any(val in pos_words for val in y_str.unique()):
            y_binary = y_str.isin(pos_words).astype(int)
        else:
            le_target = LabelEncoder()
            y_binary = pd.Series(le_target.fit_transform(y_raw), index=df.index)

    df['Churn Value'] = y_binary
    df['Churn Label'] = np.where(y_binary == 1, 'Yes', 'No')

    # Tenure
    if tenure_col and tenure_col in df.columns:
        df['Tenure Months'] = pd.to_numeric(
            df[tenure_col].astype(str).str.replace(',', '', regex=False).str.strip(),
            errors='coerce'
        )
        df['Tenure Months'] = df['Tenure Months'].fillna(
            df['Tenure Months'].median() if not df['Tenure Months'].isna().all() else 0
        )
    else:
        df['Tenure Months'] = 12

    # Monthly Charges
    if monthly_col and monthly_col in df.columns:
        df['Monthly Charges'] = pd.to_numeric(
            df[monthly_col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip(),
            errors='coerce'
        )
        df['Monthly Charges'] = df['Monthly Charges'].fillna(
            df['Monthly Charges'].median() if not df['Monthly Charges'].isna().all() else 0
        )
    else:
        df['Monthly Charges'] = 50.0

    # Total Charges
    if total_col and total_col in df.columns:
        df['Total Charges'] = pd.to_numeric(
            df[total_col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip(),
            errors='coerce'
        )
        df['Total Charges'] = df['Total Charges'].fillna(
            df['Total Charges'].median() if not df['Total Charges'].isna().all() else 0
        )
    else:
        df['Total Charges'] = df['Tenure Months'] * df['Monthly Charges']

    # Auto drop ID / meta columns
    known_id_names = [
        'customerid', 'customer_id', 'id', 'user_id', 'count', 'country', 'state', 'city',
        'zipcode', 'zip_code', 'latlong', 'lat_long', 'latitude', 'longitude',
        'churnscore', 'churn_score', 'cltv', 'churnreason', 'churn_reason', 'churn label'
    ]
    drop_cols = [target_col, 'Churn Label']
    for c in df.columns:
        cl = _clean_str(c)
        if cl in known_id_names and c not in drop_cols:
            drop_cols.append(c)
        elif df[c].nunique(dropna=True) == len(df) and df[c].dtype == 'object' and c not in drop_cols:
            drop_cols.append(c)

    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if 'Churn Value' in X_df.columns:
        X_df = X_df.drop(columns=['Churn Value'])

    # Encode remaining object columns
    for col in X_df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].fillna('Missing').astype(str))

    # Fill numeric NaNs
    for col in X_df.select_dtypes(include=[np.number]).columns:
        X_df[col] = X_df[col].fillna(X_df[col].median() if not X_df[col].isna().all() else 0)

    Y = df['Churn Value']

    mapping_info = {
        'target_col': target_col,
        'tenure_col': tenure_col,
        'monthly_col': monthly_col,
        'total_col': total_col,
        'drop_cols': drop_cols
    }

    return raw_df, df, X_df, Y, mapping_info

# Sidebar column selection & manual override
if uploaded_file is not None:
    try:
        raw_df_loaded = load_raw_dataframe(uploaded_file)
        det_target, det_tenure, det_monthly, det_total = auto_detect_columns(raw_df_loaded)

        with st.sidebar.expander("🔧 Column Configuration & Overrides", expanded=False):
            st.caption("Adjust detected dataset column roles if necessary:")
            cols_list = list(raw_df_loaded.columns)

            sel_target = st.selectbox(
                "Target (Churn) Column",
                cols_list,
                index=cols_list.index(det_target) if det_target in cols_list else 0
            )

            sel_tenure = st.selectbox(
                "Tenure Column",
                ["(Auto Detect / None)"] + cols_list,
                index=cols_list.index(det_tenure) + 1 if det_tenure in cols_list else 0
            )
            final_tenure = None if sel_tenure == "(Auto Detect / None)" else sel_tenure

            sel_monthly = st.selectbox(
                "Monthly Charges Column",
                ["(Auto Detect / None)"] + cols_list,
                index=cols_list.index(det_monthly) + 1 if det_monthly in cols_list else 0
            )
            final_monthly = None if sel_monthly == "(Auto Detect / None)" else sel_monthly

            sel_total = st.selectbox(
                "Total Charges Column",
                ["(Auto Detect / None)"] + cols_list,
                index=cols_list.index(det_total) + 1 if det_total in cols_list else 0
            )
            final_total = None if sel_total == "(Auto Detect / None)" else sel_total

        run_btn = st.sidebar.button("🚀 Run Full Pipeline", use_container_width=True)

    except Exception as e:
        st.error(f"Error loading uploaded file: {str(e)}")
        uploaded_file = None
        run_btn = False
else:
    run_btn = False

# ─────────────────────────────────────────────
# MAIN PIPELINE EXECUTION
# ─────────────────────────────────────────────
if uploaded_file and (run_btn or 'pipeline_run' in st.session_state):
    st.session_state['pipeline_run'] = True

    raw_df, df, X, Y, mapping_info = preprocess_dataset(
        raw_df_loaded, sel_target, final_tenure, final_monthly, final_total
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y if Y.nunique() > 1 else None
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
        churn_pct = round(df['Churn Value'].mean() * 100, 1)
        col3.metric("Churn Rate", f"{churn_pct}%")
        col4.metric("Non-Churn Rate", f"{round(100 - churn_pct, 1)}%")

        st.subheader("Data Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Churn Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))
            df['Churn Label'].value_counts().plot(
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
            sns.histplot(df['Tenure Months'], bins=30, kde=True, ax=ax, color='steelblue')
            ax.set_title("Tenure Months")
            plt.tight_layout()
            st.pyplot(fig)

        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader("Monthly Charges Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(df['Monthly Charges'], bins=30, kde=True, ax=ax, color='green')
            plt.tight_layout()
            st.pyplot(fig)

        with col_d:
            st.subheader("Correlation Heatmap")
            num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Value']
            num_cols = [c for c in num_cols if c in df.columns]
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            plt.tight_layout()
            st.pyplot(fig)

        st.subheader("Missing Values Summary")
        missing = raw_df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            st.success("✅ No missing values found in raw dataset!")
        else:
            st.warning(f"⚠️ {len(missing)} columns have missing values")
            st.bar_chart(missing)

    # ══════════════════════════════════════════
    # TAB 2 — MODEL TRAINING
    # ══════════════════════════════════════════
    with tab2:
        st.header("🏆 Model Training & Comparison")

        with st.spinner("Training all 3 classification models..."):
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
                
                prob = m.predict_proba(X_test)[:, 1] if hasattr(m, 'predict_proba') and len(np.unique(Y_train)) > 1 else np.zeros(len(Y_test))
                roc_val = round(roc_auc_score(Y_test, prob), 4) if len(np.unique(Y_test)) > 1 else 0.0

                results.append({
                    'Model': name,
                    'Accuracy':  round(accuracy_score(Y_test, yp), 4),
                    'Recall':    round(recall_score(Y_test, yp, zero_division=0), 4),
                    'Precision': round(precision_score(Y_test, yp, zero_division=0), 4),
                    'F1 Score':  round(f1_score(Y_test, yp, zero_division=0), 4),
                    'ROC-AUC':   roc_val
                })

        result_df = pd.DataFrame(results).sort_values('Recall', ascending=False)
        st.subheader("Model Comparison Table")
        st.dataframe(result_df.set_index('Model'), use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'ROC-AUC']
        result_df.set_index('Model')[metrics].T.plot(kind='bar', ax=ax, colormap='Set2')
        ax.set_title("All Models — Metric Benchmarks")
        ax.set_ylabel("Score")
        ax.legend(loc='lower right')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

        best_name = result_df.iloc[0]['Model']
        st.success(f"🏆 Best model by Recall: **{best_name}**")
        st.info(f"📌 Using **{model_choice}** (your selected sidebar model) for subsequent evaluation and predictions.")

    # ══════════════════════════════════════════
    # TAB 3 — EVALUATION
    # ══════════════════════════════════════════
    with tab3:
        st.header("📊 Model Evaluation")

        model = trained[model_choice]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(Y_test))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{accuracy_score(Y_test, y_pred):.2%}")
        col2.metric("Recall",    f"{recall_score(Y_test, y_pred, zero_division=0):.2%}")
        col3.metric("Precision", f"{precision_score(Y_test, y_pred, zero_division=0):.2%}")
        col4.metric("F1 Score",  f"{f1_score(Y_test, y_pred, zero_division=0):.2%}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(Y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Churn', 'Churn'],
                        yticklabels=['No Churn', 'Churn'])
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
            ax.set_title(f"Confusion Matrix — {model_choice}")
            plt.tight_layout()
            st.pyplot(fig)

        with col_b:
            st.subheader("ROC Curve")
            if len(np.unique(Y_test)) > 1:
                fpr, tpr, _ = roc_curve(Y_test, y_prob)
                auc = roc_auc_score(Y_test, y_prob)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(fpr, tpr, color='steelblue', label=f'AUC = {auc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("ROC Curve requires at least 2 classes in test set.")

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
        report = classification_report(Y_test, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    # ══════════════════════════════════════════
    # TAB 4 — SEGMENTS
    # ══════════════════════════════════════════
    with tab4:
        st.header("🗂️ Customer Segmentation")

        model = trained[model_choice]
        churn_prob_all = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(X))
        
        seg_df = pd.DataFrame({
            'Tenure Months':     df['Tenure Months'].values,
            'Monthly Charges':   df['Monthly Charges'].values,
            'Total Charges':     df['Total Charges'].values,
            'Churn Probability': churn_prob_all
        })

        scaler = StandardScaler()
        scaled = scaler.fit_transform(seg_df)

        with st.spinner("Running KMeans + Silhouette analysis..."):
            wcss, sil = [], []
            for k in range(2, 10):
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                lbl = km.fit_predict(scaled)
                wcss.append(km.inertia_)
                sil.append(silhouette_score(scaled, lbl))

        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(range(2, 10), wcss, marker='o', color='steelblue')
            ax.set_title("Elbow Method (WCSS)")
            ax.set_xlabel("k"); ax.set_ylabel("WCSS")
            plt.tight_layout(); st.pyplot(fig)
        with col_b:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(range(2, 10), sil, marker='o', color='green')
            ax.axvline(n_clusters, color='red', linestyle='--', label=f'k={n_clusters}')
            ax.set_title("Silhouette Score")
            ax.set_xlabel("k"); ax.legend()
            plt.tight_layout(); st.pyplot(fig)

        km_final = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        seg_df['Cluster'] = km_final.fit_predict(scaled)

        # Auto-name clusters by churn probability
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

        st.subheader("Cluster Profiles & Mean Feature Summary")
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
            seg_counts.plot(kind='bar', ax=ax, colormap='Set2')
            ax.set_title("Customer Count per Segment")
            ax.set_ylabel("Count")
            plt.xticks(rotation=20)
            plt.tight_layout(); st.pyplot(fig)

        st.session_state['seg_df'] = seg_df

    # ══════════════════════════════════════════
    # TAB 5 — RECOMMENDATIONS
    # ══════════════════════════════════════════
    with tab5:
        st.header("💡 Business Recommendations")

        if 'seg_df' not in st.session_state:
            st.warning("Please run segmentation in Tab 4 first.")
        else:
            seg_df = st.session_state['seg_df']
            seg_df['Revenue at Risk'] = (
                seg_df['Monthly Charges'] * seg_df['Churn Probability']
            )

            profile = seg_df.groupby('Segment').agg(
                Customers       = ('Segment', 'count'),
                Avg_Churn_Prob  = ('Churn Probability', 'mean'),
                Avg_Monthly     = ('Monthly Charges', 'mean'),
                Revenue_at_Risk = ('Revenue at Risk', 'sum')
            ).round(2)

            st.subheader("Segment Risk Profiles")
            st.dataframe(profile, use_container_width=True)

            st.subheader("💰 Monthly Revenue at Risk per Segment")
            fig, ax = plt.subplots(figsize=(7, 3))
            profile['Revenue_at_Risk'].sort_values().plot(
                kind='barh', ax=ax, colormap='Set2'
            )
            ax.set_title("Revenue at Risk per Segment ($)")
            plt.tight_layout(); st.pyplot(fig)

            st.subheader("📋 Recommended Strategic Actions")
            for seg in profile.index:
                cp = profile.loc[seg, 'Avg_Churn_Prob']
                n  = int(profile.loc[seg, 'Customers'])
                rev = profile.loc[seg, 'Revenue_at_Risk']

                if 'High' in seg:
                    color = '🔴'
                    actions = [
                        "Immediate personalized retention outreach (phone/email within 48h)",
                        "Offer targeted loyalty discounts or contract upgrade incentives",
                        "Assign dedicated account manager to address service complaints"
                    ]
                elif 'Medium' in seg:
                    color = '🟡'
                    actions = [
                        "Send automated customer satisfaction survey to identify pain points",
                        "Offer value-add add-on services (security, tech support, streaming)",
                        "Enroll in customer loyalty program and referral rewards"
                    ]
                else:
                    color = '🟢'
                    actions = [
                        "Upsell to premium long-term plan tiers",
                        "Incentivize multi-line family or business referrals",
                        "Reward continued tenure with anniversary bonus perks"
                    ]

                with st.expander(f"{color} {seg} — {n} customers | Churn Prob: {cp:.0%} | Revenue at Risk: ${rev:,.0f}"):
                    for a in actions:
                        st.write(f"→ {a}")

            st.subheader("📌 Executive Summary Table")
            profile['Action'] = profile.index.map(
                lambda s: 'IMMEDIATE RETENTION' if 'High' in s else ('MONITOR & ENGAGE' if 'Medium' in s else 'UPSELL & REWARD')
            )
            st.dataframe(profile, use_container_width=True)

else:
    st.info("👈 Upload your Excel or CSV file in the sidebar and click **Run Full Pipeline** to begin analysis.")
    st.markdown("""
    ### What this dashboard does:
    | Tab | Description |
    |-----|-------------|
    | 🔍 EDA | Automatically explores dataset features, distributions, correlations, and missing values |
    | 🏆 Model | Trains 3 classification models (Random Forest, Gradient Boosting, Logistic Regression) & compares metrics |
    | 📊 Evaluation | Displays confusion matrix, ROC curve, feature importances, and classification report |
    | 🗂️ Segments | Runs KMeans clustering with automated Elbow & Silhouette score optimal k analysis |
    | 💡 Recommendations | Computes revenue at risk and provides targeted business retention strategies |
    """)
