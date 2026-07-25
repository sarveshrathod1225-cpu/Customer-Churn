# 📊 Customer Segmentation and Churn Analysis 

> **An advanced ML analytics platform with multiple implementation approaches - from interactive Jupyter notebooks to production-ready agentic dashboards!**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![ML](https://img.shields.io/badge/ML-ScikitLearn-orange.svg)](https://scikit-learn.org)

---

## 🎯 **Choose Your Adventure!**

| 🚀 **Approach** | 📝 **Description** | 🎮 **Best For** | ⚡ **Quick Start** |
|:---|:---|:---|:---|
| **🤖 Agentic Dashboard** | Fully automated Streamlit app with one-click analysis | Production use, demos, non-technical users | [Jump to Dashboard](#-agentic-dashboard---production-ready) |
| **📊 Jupyter Notebooks** | Step-by-step analysis with detailed explanations | Learning, experimentation, customization | [Open Notebooks](#-jupyter-notebooks---deep-dive-analysis) |
| **🐳 Docker Container** | Containerized deployment for any environment | Cloud deployment, scalability, DevOps | [Docker Setup](#-docker-deployment---one-command-launch) |

---

## 📁 **Complete Project Structure**

```
customer_segmentation_and_churn_analysis/
├── 🤖 churn_app/                           ← AGENTIC DASHBOARD
│   ├── app.py                              ← Main Streamlit application
│   ├── requirements.txt                    ← Dependencies
│   ├── Dockerfile & docker-compose.yml        ← Container setup
│   ├── data/Telco_customer_churn.xlsx         ← Dataset
│   ├── dashboard_notebook_version.ipynb        ← Dashboard in notebook format
│   └── screenshots/                        ← Demo images
├── 📂 notebooks/                            ← ANALYSIS NOTEBOOKS
│   ├── 01_basic_analysis/
│   │   └── customer_churn_analysis_complete.ipynb  ← Complete EDA + ML + Segmentation
│   ├── 02_advanced_techniques/
│   │   └── advanced_ml_techniques.ipynb         ← Hyperparameter tuning + Ensembles
│   └── 03_automated_pipeline/
│       └── automated_churn_pipeline.ipynb       ← End-to-end automation
└── 📋 Telco_customer_churn.xlsx              ← Raw dataset
```

---

## 🤖 **Agentic Dashboard - Production Ready**

> **⚡ One-click customer segmentation and churn analysis with zero coding required!** Upload data → Click button → Get insights!

### 🎮 **What Makes It "Agentic"?**
- **🤖 Autonomous**: Automatically segments customers and predicts churn
- **🧠 Intelligent**: Compares 3 ML models and picks optimal hyperparameters  
- **🔄 Adaptive**: Dynamic customer segmentation with auto-naming
- **💡 Actionable**: Generates targeted retention strategies automatically

### 🚀 **Quick Launch Options**

<details>
<summary><b>🔥 Option 1: Local Run (Fastest)</b></summary>

```bash
# Navigate to dashboard
cd churn_app/

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py

# Open browser: http://localhost:8501
```
</details>

<details>
<summary><b>🐳 Option 2: Docker (Recommended)</b></summary>

```bash
# One command launch
cd churn_app/
docker-compose up --build

# Open browser: http://localhost:8501
# Stop: docker-compose down
```
</details>

<details>
<summary><b>🔧 Option 3: Manual Docker</b></summary>

```bash
cd churn_app/

# Build image
docker build -t churn-dashboard .

# Run container
docker run -p 8501:8501 churn-dashboard
```
</details>

---

## 📊 Dashboard Features

| Tab | What it does |
|-----|-------------|
| 🔍 EDA | Auto explores data — distributions, correlations, missing values |
| 🏆 Model | Trains Random Forest, Gradient Boosting, Logistic Regression and compares |
| 📊 Evaluation | Confusion matrix, ROC curve, feature importances, classification report |
| 🗂️ Segments | KMeans clustering with Elbow + Silhouette for optimal k |
| 💡 Recommendations | Revenue at risk + business retention strategies per segment |

## 📊 **Jupyter Notebooks - Deep Dive Analysis**

> **🎯 Perfect for learning, experimentation, and custom analysis!** 

| 📖 **Notebook** | 📝 **Focus** | 🎮 **Level** | 🔗 **Quick Access** |
|:---|:---|:---:|:---|
| **Customer Churn Analysis Complete** | End-to-end analysis: EDA → ML Models → Customer Segmentation | 🟢 Beginner | [Open Complete Analysis](../notebooks/01_basic_analysis/customer_churn_analysis_complete.ipynb) |
| **Advanced ML Techniques** | Hyperparameter tuning, ensemble methods, model optimization | 🟡 Advanced | [Open Advanced Techniques](../notebooks/02_advanced_techniques/advanced_ml_techniques.ipynb) |
| **Automated Churn Pipeline** | Fully automated ML pipeline with minimal human intervention | 🔴 Expert | [Open Automated Pipeline](../notebooks/03_automated_pipeline/automated_churn_pipeline.ipynb) |
| **Dashboard Notebook Version** | Streamlit dashboard logic in notebook format for customization | 🟠 Custom | [Open Dashboard Logic](dashboard_notebook_version.ipynb) |

### 📚 **Learning Path Recommendation:**
```
🎆 Start Here → 🟢 Complete Analysis → 🟡 Advanced Techniques → 🔴 Automated Pipeline → 🟠 Custom Dashboard
```

### 🚀 **Launch Jupyter Environment**

<details>
<summary><b>💻 Local Jupyter Setup</b></summary>

```bash
# Install Jupyter (if not already installed)
pip install jupyter notebook

# Navigate to notebooks folder
cd notebooks/

# Launch Jupyter
jupyter notebook

# Opens in browser automatically
# Start with: 01_basic_analysis/customer_churn_analysis_complete.ipynb
```
</details>

<details>
<summary><b>🌐 JupyterLab (Enhanced Experience)</b></summary>

```bash
# Install JupyterLab
pip install jupyterlab

# Navigate to project root
cd customer_segmentation_and_churn_analysis/

# Launch JupyterLab
jupyter lab

# Enhanced interface with file explorer, terminal, etc.
# Navigate to notebooks/ folder in the sidebar
```
</details>

<details>
<summary><b>🐍 Anaconda Users</b></summary>

```bash
# Create conda environment
conda create -n customer-churn-analysis python=3.11
conda activate customer-churn-analysis

# Install packages
conda install pandas numpy matplotlib seaborn scikit-learn jupyter
pip install streamlit openpyxl

# Navigate and launch
cd notebooks/01_basic_analysis/
jupyter notebook
```
</details>

## 🐳 **Docker Deployment - One Command Launch**

> **🚀 Perfect for production, cloud deployment, and ensuring consistency across environments!**

### 🎯 **Why Docker?**
- ✅ **Consistency**: Same environment everywhere
- ✅ **Scalability**: Easy cloud deployment 
- ✅ **Isolation**: No dependency conflicts
- ✅ **Portability**: Runs on any Docker-enabled system

### 🚀 **Super Quick Deploy**

```bash
# Clone & Run (One-liner for fresh setup)
git clone <your-repo> && cd customer_segmentation_and_churn_analysis/churn_app && docker-compose up --build
```

<details>
<summary><b>🌍 Cloud Deployment Examples</b></summary>

**AWS ECS:**
```bash
# Build for AWS
docker build -t churn-dashboard .
docker tag churn-dashboard:latest <account>.dkr.ecr.<region>.amazonaws.com/churn-dashboard:latest
```

**Google Cloud Run:**
```bash
# Deploy to Cloud Run
gcloud builds submit --tag gcr.io/<project>/churn-dashboard
gcloud run deploy --image gcr.io/<project>/churn-dashboard --platform managed
```

**Azure Container Instances:**
```bash
# Deploy to Azure
az container create --resource-group myResourceGroup --name churn-dashboard --image <your-registry>/churn-dashboard:latest
```
</details>

---

## 🖼️ Demo Screenshots

<details>
<summary><b>🔍 Click to View Dashboard Screenshots</b></summary>

### EDA Dashboard - Auto Data Exploration
![EDA Overview](churn_app/Screenshot%202026-06-24%20175045.png)
*Automatic data profiling, missing value analysis, and correlation discovery*

### Model Training & Comparison - AI Model Selection
![Model Comparison](churn_app/Screenshot%202026-06-24%20175105.png)
*Side-by-side performance metrics for Random Forest, Gradient Boosting & Logistic Regression*

### Customer Segmentation - Intelligent Clustering
![Customer Segments](churn_app/Screenshot%202026-06-24%20175120.png)
*Automated KMeans clustering with optimal k-selection and segment profiling*

### Business Recommendations - Actionable Insights
![Recommendations](churn_app/Screenshot%202026-06-24%20175140.png)
*Revenue-at-risk analysis and targeted retention strategies per customer segment*

</details>

## 🎆 **What Makes This Special?**

<div align="center">

| 🅰️ **Agentic Dashboard** | 🅱️ **Jupyter Notebooks** |
|:---:|:---:|
| ⚡ **Zero-code required** | 🔍 **Full transparency** |
| 🤖 **Automated insights** | 🎮 **Customizable analysis** |
| 📨 **Business-ready reports** | 📚 **Educational content** |
| 🚀 **One-click deployment** | 🔧 **Research & experimentation** |

</div>

### 🎯 **Key Differentiators:**
- **🤖 Intelligent Automation**: Not just a static dashboard - adapts to your data
- **📈 Business Focus**: Revenue impact analysis, not just technical metrics
- **🔄 Multiple Approaches**: Choose based on your needs (production vs research)
- **🌐 Production Ready**: Docker containerization for enterprise deployment
- **📚 Educational**: Complete learning path from basics to advanced

---

## 🛠️ Tech Stack

- **Python 3.11**
- **Streamlit** — Dashboard UI
- **Scikit-learn** — ML models + clustering
- **Pandas / NumPy** — Data processing
- **Matplotlib / Seaborn** — Visualisations
- **Docker** — Containerisation

---

## 🕰️ **Quick Start Guide**

<div align="center">

### 🎆 **30-Second Demo**
```bash
# The fastest way to see it in action!
cd churn_app && python -m streamlit run app.py
# → Upload Excel → Click "Run Pipeline" → Explore 5 tabs of insights!
```

</div>

<details>
<summary><b>🎯 Step-by-Step Walkthrough</b></summary>

**For the Agentic Dashboard:**
1. 📤 **Upload** `Telco_customer_churn.xlsx` via sidebar (or use bundled dataset)
2. 🎲 **Choose** your preferred ML model from dropdown
3. 🔢 **Adjust** number of customer segments (2-8)
4. 🚀 **Click** "Run Full Pipeline" to execute complete analysis
5. 🔍 **Navigate** through 5 tabs:
   - **EDA**: Data exploration and visualization
   - **Model**: ML model training and comparison  
   - **Evaluation**: Performance metrics and analysis
   - **Segments**: Customer clustering and profiles
   - **Recommendations**: Actionable business insights

**For Jupyter Analysis:**
1. 💻 **Launch** `jupyter notebook` or `jupyter lab`
2. 📂 **Open** desired notebook based on your level
3. ▶️ **Run** cells sequentially (Shift+Enter)
4. 🎮 **Experiment** with parameters and add custom analysis
5. 💾 **Save** your customized version

</details>

## 🔧 **Tech Stack & Architecture**

<div align="center">

| Layer | Technology | Purpose |
|:---:|:---:|:---|
| **🌐 Frontend** | Streamlit | Interactive dashboard UI |
| **🧠 ML Engine** | Scikit-learn | Models, clustering, preprocessing |
| **📈 Data Processing** | Pandas, NumPy | Data manipulation & analysis |
| **🎨 Visualization** | Matplotlib, Seaborn | Charts, plots, heatmaps |
| **🐳 Deployment** | Docker, Docker Compose | Containerization & orchestration |
| **📚 Development** | Jupyter | Interactive development & research |

</div>

### 🎨 **ML Pipeline Architecture:**
```
📊 Raw Data → 🧼 Preprocessing → 🎯 Feature Engineering → 🤖 Churn Prediction → 🗂️ Customer Segmentation → 💡 Insights
     │              │                    │                      │                     │                         │
   Excel/CSV    • Missing values    • Encoding         • Random Forest       • KMeans Clustering       • Business
   Upload       • Data types       • Scaling          • Gradient Boost      • Optimal K Selection     • Actions
                • Outliers         • Selection        • Logistic Reg        • Segment Profiles        • Revenue
```

## 🎁 **What You'll Get**

<div align="center">

| 📊 **Analytics** | 🎮 **Interactivity** | 📈 **Business Value** |
|:---:|:---:|:---:|
| • Churn probability scores | • Real-time model comparison | • Revenue-at-risk analysis |
| • Customer segment profiles | • Interactive visualizations | • Targeted retention strategies |
| • Feature importance ranking | • Parameter tuning controls | • ROI-focused recommendations |
| • Performance benchmarks | • Dynamic clustering | • Executive-ready reports |

</div>

---

## 📞 **Support & Contributing**

<div align="center">

**Found this helpful? ⭐ Star the repo!**

[🐛 Report Issues](../../issues) | [💡 Feature Requests](../../issues) | [💬 Discussions](../../discussions)

</div>

<details>
<summary><b>🤝 Contributing Guidelines</b></summary>

1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to branch (`git push origin feature/AmazingFeature`)
5. **Open** Pull Request

**Areas for contribution:**
- 🎨 Additional visualizations
- 🤖 New ML models/algorithms  
- 📈 Enhanced business metrics
- 🌐 Cloud deployment templates
- 📚 Documentation improvements

</details>

---

<div align="center">

**🚀 Ready to unlock customer insights and boost retention? Choose your path above and get started!**

*Made with ❤️ for data scientists, analysts, and business teams working on customer segmentation and churn analysis*

</div>
