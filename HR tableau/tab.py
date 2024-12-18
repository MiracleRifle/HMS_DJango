!pip install streamlit
!pip install pyngrok
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class HRAnalyticsDashboard:
    def __init__(self, db_path):
        # Database connection and data loading
        self.conn = sqlite3.connect('/content/datalog.sqlite')
        self.df = pd.read_sql_query("SELECT * FROM your_table_name", self.conn)

        # Comprehensive data preprocessing
        self.preprocess_data()

    def preprocess_data(self):
        # Advanced feature engineering
        self.df['Years of Experience'] = self.df['TotalWorkingYears']
        self.df['Age_Band'] = pd.cut(self.df['Age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['Young', 'Early Career', 'Mid Career', 'Experienced', 'Senior']
        )

        # Performance potential calculation
        self.df['Performance Potential'] = (
            self.df['PerformanceRating'] *
            (1 + self.df['RelationshipSatisfaction']/100)
        )

        # Handle missing values
        self.df.fillna({
            'Performance Score': self.df['PerformanceRating'].median(),
            'Engagement Survey': self.df['RelationshipSatisfaction'].median()
        }, inplace=True)

    def performance_analysis(self):
        # Interactive Plotly performance visualization
        fig = px.box(
            self.df,
            x='Department',
            y='Performance Score',
            color='Gender',
            title='Performance Scores by Department and Gender'
        )
        st.plotly_chart(fig)

    def attrition_prediction(self):
        # Advanced predictive modeling
        features = [
            'Age', 'TotalWorkingYears', 'YearsAtCompany',
            'JobSatisfaction', 'PerformanceRating',
            'EnvironmentSatisfaction', 'RelationshipSatisfaction',
            'DistanceFromHome',
            'MonthlyIncome'
        ]

        X = self.df[features]
        y = (self.df['Attrition'] == 'Yes').astype(int)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)


        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # Multiple model comparison
        models = {
            'Logistic Regression': LogisticRegression(solver='liblinear', max_iter=5000),
            'Random Forest': RandomForestClassifier()
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            results[name] = model.score(X_test, y_test)

        return results

    def interactive_dashboard(self):
        # Streamlit dashboard with multiple sections
        st.title('Comprehensive HR Analytics Dashboard')

        # Sidebar filters
        st.sidebar.header('Dashboard Filters')
        selected_dept = st.sidebar.multiselect(
            'Select Departments',
            self.df['Department'].unique()
        )

        # Performance Section
        st.header('Performance Insights')
        perf_fig = px.scatter(
            self.df,
            x='Engagement Survey',
            y='Performance Score',
            color='Department',
            title='Performance vs Engagement'
        )
        st.plotly_chart(perf_fig)

        # Attrition Section
        st.header('Attrition Analysis')
        attrition_fig = px.pie(
            self.df[self.df['Attrition'] == 'Yes'],
            names='Attrition Reason',
            title='Reasons for Attrition'
        )
        st.plotly_chart(attrition_fig)

        # Salary Insights
        st.header('Salary Distribution')
        salary_fig = px.box(
            self.df,
            x='Department',
            y='Monthly Income',
            color='Gender',
            title='Salary Distribution by Department and Gender'
        )
        st.plotly_chart(salary_fig)

# Main execution
def main():
    # Initialize dashboard
    hr_dashboard = HRAnalyticsDashboard('employee_database.sqlite')

    # Run predictive analysis
    prediction_results = hr_dashboard.attrition_prediction()
    print("Model Performance:", prediction_results)

    # Optional: Run Streamlit dashboard
    # streamlit run script.py

if __name__ == "__main__":
    main()