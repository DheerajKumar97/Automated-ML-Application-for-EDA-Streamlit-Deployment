# Machine Learning Application for Automated EDA

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

A comprehensive web-based application that automates exploratory data analysis (EDA) and machine learning model building, enabling data analysts and business users to gain actionable insights without writing code.

## 🎯 Business Problem

Data analysts and business users often spend excessive time on repetitive tasks:
- **Manual data exploration** consuming valuable analysis time
- **Repetitive data cleaning and type checking** prone to human error
- **Time-intensive visualization creation** for pattern discovery
- **Initial model development** requiring significant coding effort

These manual processes increase the time to actionable insights and make diagnostics error-prone, delaying critical business decisions.

## 💡 Solution Approach

This Streamlit-based application provides an interactive, no-code interface for:

- **Automated Data Inspection**: Intelligent type conversion, missing value detection, and data quality assessment
- **Interactive Visualizations**: One-click generation of histograms, correlation heatmaps, pair plots, and count plots
- **Statistical Analysis**: Correlation analysis, outlier detection, and distribution assessment
- **Quick Model Building**: Train and evaluate multiple classification models instantly
- **User-Friendly Interface**: Clean, intuitive design requiring zero coding knowledge

## ✨ Features

### 📊 General EDA
- **Smart Data Loading**: Automatic CSV parsing with intelligent type detection
- **Data Type Conversion**: Convert columns to appropriate types (numeric, datetime, boolean)
- **Missing Value Analysis**: Identify and handle missing data
- **Statistical Summaries**: Comprehensive tabulation of data characteristics
- **Visualization Suite**:
  - Correlation heatmaps
  - Pair plots for multivariate analysis
  - Histograms with KDE curves
  - Count plots for categorical data
- **Column Information Dashboard**: Quick overview of variable types and counts

### 📈 Linear Model EDA
- **Q-Q Plots**: Assess normality of distributions
- **Outlier Detection**: IQR-based outlier identification
- **Distribution Analysis**: Evaluate data suitability for linear models

### 🤖 Model Building
Supports multiple classification algorithms:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Naive Bayes (Gaussian)**
- **XGBoost Classifier**

Each model provides detailed classification reports with precision, recall, and F1-scores.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/DheerajKumar97/automated-eda-ml.git
cd automated-eda-ml
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Required Dependencies

Create a `requirements.txt` file with the following:

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.3.0
xgboost>=2.0.0
statsmodels>=0.14.0
Pillow>=10.0.0
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. General EDA Workflow

1. **Upload Data**: Click "Browse files" and select your CSV file
2. **Enable Smart Conversion** (optional): Check "Intelligently Convert Column Types" for automatic type detection
3. **Select Analysis Options**: Use the sidebar to enable various analysis features:
   - Show data types
   - Display missing values
   - Generate correlation heatmaps
   - Create pair plots
   - View histograms and count plots

### 2. Linear Model EDA

1. **Upload Data**: Load your CSV file
2. **Select Column**: Choose a numeric column for analysis
3. **Generate Q-Q Plot**: Assess normality visually
4. **Detect Outliers**: Identify potential outliers using IQR method

### 3. Model Building

1. **Upload Data**: Load your CSV file
2. **Select Features**: Choose columns (ensure target variable is last)
3. **Train Models**: Click on desired model buttons
4. **Review Results**: Examine classification reports for model performance

## 🏗️ Project Structure

```
automated-eda-ml/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── cover.jpg                   # Application cover image (optional)
├── README.md                   # Project documentation
│
└── classes/
    ├── DataFrame_Loader        # Data loading and type conversion
    ├── EDA_Dataframe_Analysis  # Statistical analysis and visualization
    ├── Attribute_Information   # Data profiling utilities
    └── Data_Base_Modelling     # Machine learning model wrapper
```

## 🔧 Technical Challenges & Solutions

### Challenge 1: Robust Type Conversion
**Problem**: CSV files contain diverse data types that aren't always correctly inferred

**Solution**: Implemented intelligent type conversion that:
- Attempts datetime parsing with multiple formats
- Falls back to numeric conversion
- Detects boolean values
- Maintains data integrity throughout the process

### Challenge 2: Dynamic Data Validation
**Problem**: Plotting and modeling errors when columns are missing or unsuitable

**Solution**: 
- Pre-validation of column types before operations
- Clear warning messages when operations can't be performed
- Graceful handling of empty or invalid selections

### Challenge 3: Matplotlib/Seaborn Integration
**Problem**: Figure management and display in Streamlit's reactive environment

**Solution**:
- Proper backend configuration (`plt.switch_backend("Agg")`)
- Explicit figure closing after each plot
- Use of `st.pyplot(plt.gcf())` for consistent rendering

### Challenge 4: User Experience
**Problem**: Providing informative feedback without overwhelming users

**Solution**:
- Progressive disclosure of options via sidebar
- Contextual warnings and success messages
- Clear labeling and intuitive flow

## 🎨 Key Components

### DataFrame_Loader
Handles data ingestion and intelligent type conversion, ensuring data is properly formatted for analysis.

### EDA_Dataframe_Analysis
Core analysis engine providing:
- Statistical summaries
- Correlation analysis
- Visualization generation
- Outlier detection
- PCA transformation

### Attribute_Information
Generates comprehensive data profiling reports showing variable types, counts, and quality metrics.

### Data_Base_Modelling
Unified interface for training and evaluating classification models with consistent output formatting.

## 📊 Sample Workflows

### Workflow 1: Quick Data Quality Check
```
1. Upload CSV
2. Enable "Column Information"
3. Enable "Show Missing Values"
4. Enable "Tabulation Summary"
→ Get comprehensive data quality report
```

### Workflow 2: Feature Correlation Analysis
```
1. Upload CSV
2. Enable "Show HeatMap"
3. Select relevant numeric columns
4. Analyze correlation patterns
→ Identify multicollinearity and relationships
```

### Workflow 3: Rapid Model Prototyping
```
1. Upload CSV
2. Navigate to "Model Building"
3. Select features and target
4. Test multiple models
5. Compare classification reports
→ Identify best-performing algorithm
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Future Enhancements

- [ ] Add regression model support
- [ ] Implement feature engineering tools
- [ ] Add data export functionality
- [ ] Include time series analysis
- [ ] Add model persistence (save/load)
- [ ] Implement A/B testing framework
- [ ] Add custom visualization builder
- [ ] Support for multiple file formats (Excel, JSON, Parquet)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**DHEERAJ KUMAR K**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/dheerajkumar1997/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/DheerajKumar97?tab=repositories)
[![Website](https://img.shields.io/badge/Website-FF7139?style=for-the-badge&logo=firefox-browser&logoColor=white)](https://dheeraj-kumar-k.lovable.app/)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [scikit-learn](https://scikit-learn.org/)
- Visualizations by [Seaborn](https://seaborn.pydata.org/) and [Plotly](https://plotly.com/)

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Contact via LinkedIn
- Visit the project website

---

**Made with ❤️ for the data science community**
