import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from PIL import Image
import statsmodels.api as sm

plt.switch_backend("Agg")


class DataFrame_Loader:
    def read_csv(self, data):
        df = pd.read_csv(data)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            try:
                df[col] = pd.to_datetime(df[col], errors='raise', dayfirst=True)
            except Exception:
                pass
        return df

    def intelligent_type_conversion(self, df):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
                continue
            try:
                df[col] = pd.to_datetime(df[col], errors='raise', dayfirst=True)
                continue
            except Exception:
                pass
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
                continue
            except Exception:
                pass
            if df[col].dropna().isin(['True', 'False', 'true', 'false', True, False]).all():
                df[col] = df[col].astype('bool')
        return df


class EDA_Dataframe_Analysis:
    def show_dtypes(self, x):
        return x.dtypes

    def show_columns(self, x):
        return x.columns

    def Show_Missing(self, x):
        return x.isna().sum()

    def Tabulation(self, x):
        table = pd.DataFrame(x.dtypes, columns=['dtypes']).reset_index().rename(columns={'index': 'Name'})
        table['No of Missing'] = x.isnull().sum().values
        table['No of Uniques'] = x.nunique().values
        table['Percent of Missing'] = (x.isnull().sum().values / x.shape[0]) * 100
        return table

    def Numerical_variables(self, x):
        return x.select_dtypes(include=[np.number])

    def categorical_variables(self, x):
        return x.select_dtypes(exclude=[np.number])

    def impute(self, x):
        return x.dropna()

    def Show_pearsonr(self, x, y):
        return pearsonr(x, y)

    def Show_spearmanr(self, x, y):
        return spearmanr(x, y)

    def plotly(self, a, x, y):
        fig = px.scatter(a, x=x, y=y)
        st.plotly_chart(fig)

    def Show_DisPlot(self, x):
        plt.figure(figsize=(8, 5))
        sns.histplot(x, bins=25, kde=True)
        st.pyplot(plt.gcf())
        plt.close()

    def Show_CountPlot(self, x):
        plt.figure(figsize=(8, 5))
        sns.countplot(x=x)
        st.pyplot(plt.gcf())
        plt.close()

    def Show_PairPlot(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_columns = st.multiselect(
            "Select numeric columns for PairPlot",
            options=numeric_cols,
            default=numeric_cols if len(numeric_cols) <= 5 else numeric_cols[:5]
        )
        if len(selected_columns) < 2:
            st.warning("Please select at least two numeric columns for PairPlot.")
            return
        sns.pairplot(df[selected_columns])
        st.pyplot(plt.gcf())
        plt.close()

    def Show_HeatMap(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available for correlation heatmap.")
            return
        selected_columns = st.multiselect(
            "Select numeric columns for HeatMap",
            options=numeric_cols,
            default=numeric_cols
        )
        if not selected_columns:
            st.warning("Please select at least one numeric column.")
            return
        corr_df = df[selected_columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt.gcf())
        plt.close()

    def PCA(self, x):
        numeric_x = x.select_dtypes(include=[np.number])
        pca = PCA(n_components=min(8, numeric_x.shape[1]))
        principlecomponents = pca.fit_transform(numeric_x)
        return pd.DataFrame(principlecomponents)

    def outlier(self, x):
        q1 = x.quantile(.25)
        q3 = x.quantile(.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return x[(x < low) | (x > high)]

    def check_cat_relation(self, x, y, confidence_interval):
        cross_table = pd.crosstab(x, y)
        stat, p, dof, expected = chi2_contingency(cross_table)
        alpha = 1 - confidence_interval
        return {"chi2": stat, "p_value": p, "alpha": alpha, "related": p <= alpha}


class Attribute_Information:
    def Column_information(self, data):
        data_info = pd.DataFrame({
            "value": [
                data.shape[0],
                data.shape[1],
                data._get_numeric_data().shape[1],
                data.select_dtypes(include='category').shape[1],
                data.select_dtypes(include='object').shape[1],
                data.select_dtypes(include='bool').shape[1],
                data.select_dtypes(include='datetime64').shape[1],
                data.loc[:, data.apply(pd.Series.nunique) == 1].shape[1],
            ]
        }, index=[
            'No of observation',
            'No of Variables',
            'No of Numerical Variables',
            'No of Factor Variables',
            'No of Categorical Variables',
            'No of Logical Variables',
            'No of Date Variables',
            'No of zero variance variables'
        ])
        return data_info


class Data_Base_Modelling:
    def IMpupter(self, x):
        numeric_x = x.select_dtypes(include=[np.number])
        imp_mean = IterativeImputer(random_state=0)
        x_imp = imp_mean.fit_transform(numeric_x)
        return pd.DataFrame(x_imp, columns=numeric_x.columns)

    def Logistic_Regression(self, x_train, y_train, x_test, y_test):
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train, y_train)
        return classification_report(y_test, model.predict(x_test))

    def Decision_Tree(self, x_train, y_train, x_test, y_test):
        model = DecisionTreeClassifier()
        model.fit(x_train, y_train)
        return classification_report(y_test, model.predict(x_test))

    def RandomForest(self, x_train, y_train, x_test, y_test):
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        return classification_report(y_test, model.predict(x_test))

    def naive_bayes(self, x_train, y_train, x_test, y_test):
        model = GaussianNB()
        model.fit(x_train, y_train)
        return classification_report(y_test, model.predict(x_test))

    def XGb_classifier(self, x_train, y_train, x_test, y_test):
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(x_train, y_train)
        return classification_report(y_test, model.predict(x_test))


st.set_page_config(page_title="Automated EDA & ML", layout="wide")
st.title("Machine Learning Application for Automated EDA")
try:
    st.image(Image.open("cover.jpg"))
except:
    st.warning("No cover image found.")

st.info(
    """
    Created and maintained by **DHEERAJ KUMAR K** 👉 [LinkedIn](https://www.linkedin.com/in/dheerajkumar1997/) | 
    [GitHub](https://github.com/DheerajKumar97?tab=repositories) | 
    [Website](https://dheeraj-kumar-k.lovable.app/)
    """
)

activities = ["General EDA", "EDA For Linear Models", "Model Building for Classification Problem"]
choice = st.sidebar.selectbox("Select Activities", activities)

load = DataFrame_Loader()
dataframe = EDA_Dataframe_Analysis()
info = Attribute_Information()
model = Data_Base_Modelling()

def main():
    if choice == "General EDA":
        st.subheader("Exploratory Data Analysis")
        data = st.file_uploader("Upload a CSV file", type=["csv"])
        df = None
        if data is not None:
            df = load.read_csv(data)
            if st.sidebar.checkbox("Intelligently Convert Column Types"):
                df = load.intelligent_type_conversion(df)
                st.success("Columns converted intelligently")
                st.dataframe(df.head())
                st.write(df.dtypes)
            else:
                st.dataframe(df.head())
            st.success("Data loaded successfully")
            st.sidebar.header("EDA Options")
            if st.sidebar.checkbox("Show Dtypes"):
                st.subheader("Data Types of Each Column")
                st.write(dataframe.show_dtypes(df))
            if st.sidebar.checkbox("Show Columns"):
                st.subheader("Show Columns")
                st.write(dataframe.show_columns(df))
            if st.sidebar.checkbox("Show Missing Values"):
                st.subheader("Missing Values in Each Column")
                st.write(dataframe.Show_Missing(df))
            if st.sidebar.checkbox("Column Information"):
                st.subheader("Column Information")
                st.write(info.Column_information(df))
            if st.sidebar.checkbox("Tabulation Summary"):
                st.subheader("Tabulation Summary")
                st.write(dataframe.Tabulation(df))
            if st.sidebar.checkbox("Drop Missing Rows (Impute)"):
                st.subheader("Drop Null or NA Rows")
                st.dataframe(dataframe.impute(df))
            if st.sidebar.checkbox("Show HeatMap"):
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns available for heatmap.")
                else:
                    st.subheader("Correlation Heatmap")
                    dataframe.Show_HeatMap(df)
            if st.sidebar.checkbox("Show PairPlot"):
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns available for pairplot.")
                else:
                    st.subheader("Pair Plot")
                    dataframe.Show_PairPlot(df)
            if st.sidebar.checkbox("Histogram Plot"):
                col = st.selectbox("Select column for histogram", df.columns, key="hist_col")
                dataframe.Show_DisPlot(df[col].dropna())
            if st.sidebar.checkbox("Count Plot"):
                cat_col = st.selectbox("Select categorical column", df.columns, key="count_col")
                dataframe.Show_CountPlot(df[cat_col])

    elif choice == "EDA For Linear Models":
        st.subheader("EDA For Linear Models")
        data = st.file_uploader("Upload a CSV file", type=["csv"], key="linear")
        df = None
        if data is not None:
            df = load.read_csv(data)
            st.dataframe(df.head())
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Select numeric column for QQ Plot", numeric_cols, key="qq_col")
                if st.checkbox("Show QQ Plot", key="qq"):
                    col_data = df[col].dropna()
                    if not col_data.empty:
                        fig = sm.qqplot(col_data, line='45')
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.warning("Selected column has no data after removing missing values.")
            else:
                st.warning("No numeric columns available for QQ Plot or Outlier Detection.")
        if df is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                out_col = st.selectbox("Select numeric column for Outlier Detection", numeric_cols, key="out_col")
                if st.checkbox("Show Outliers", key="outliers"):
                    col_data = df[out_col].dropna()
                    if not col_data.empty:
                        outliers = dataframe.outlier(col_data)
                        if outliers.empty:
                            st.info(f"No outliers detected in column '{out_col}'.")
                        else:
                            st.write(f"Outliers in column '{out_col}':")
                            st.write(outliers)
                    else:
                        st.warning(f"The selected column '{out_col}' has no data after removing missing values.")
            else:
                st.warning("No numeric columns available for Outlier Detection.")

    elif choice == "Model Building for Classification Problem":
        st.subheader("Model Building")
        data = st.file_uploader("Upload a CSV file", type=["csv"], key="model")
        df = None
        if data is not None:
            df = load.read_csv(data)
            st.dataframe(df.head())
            selected_columns = st.multiselect("Select Columns (target last)", df.columns)
            if selected_columns:
                df_selected = df[selected_columns]
                x = df_selected.iloc[:, :-1]
                y = df_selected.iloc[:, -1]
                from sklearn.model_selection import train_test_split
                x_numeric = x.select_dtypes(include=[np.number])
                if not x_numeric.empty:
                    x_train, x_test, y_train, y_test = train_test_split(x_numeric, y, random_state=0)
                    st.write("Choose a Model to Train:")
                    if st.button("Logistic Regression"):
                        st.text(model.Logistic_Regression(x_train, y_train, x_test, y_test))
                    if st.button("Decision Tree"):
                        st.text(model.Decision_Tree(x_train, y_train, x_test, y_test))
                    if st.button("Random Forest"):
                        st.text(model.RandomForest(x_train, y_train, x_test, y_test))
                    if st.button("Naive Bayes"):
                        st.text(model.naive_bayes(x_train, y_train, x_test, y_test))
                    if st.button("XGBoost"):
                        st.text(model.XGb_classifier(x_train, y_train, x_test, y_test))
                else:
                    st.warning("No numeric columns selected as predictors!")

if __name__ == "__main__":
    main()
