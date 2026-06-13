import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_mock_data(filename='ecommerce_sales_data.csv'):
    # Generate a synthetic dataset to simulate an e-commerce environment
    np.random.seed(101)
    n_samples = 1000
    
    data = {
        'CustomerID': range(1001, 1001 + n_samples),
        'Age': np.random.randint(18, 70, n_samples),
        'Income': np.random.normal(65000, 20000, n_samples).round(2),
        'Website_Visits': np.random.poisson(15, n_samples),
        'Time_Spent_Min': np.random.normal(30, 10, n_samples).round(2),
        'Total_Spent': np.zeros(n_samples) # Will calculate below based on features
    }
    df = pd.DataFrame(data)

    # Inject correlations to make EDA meaningful
    # Total spent increases significantly with Income and Time_Spent_Min
    df['Total_Spent'] = (df['Income'] * 0.015 + df['Time_Spent_Min'] * 8 + df['Age'] * 1.5 + np.random.normal(0, 150, n_samples)).round(2)
    
    # Introduce some missing values to simulate real-world data
    df.loc[np.random.choice(df.index, 45, replace=False), 'Income'] = np.nan
    df.loc[np.random.choice(df.index, 20, replace=False), 'Time_Spent_Min'] = np.nan

    df.to_csv(filename, index=False)
    print(f"Mock dataset generated: {filename}")
    return filename

def perform_eda(filepath, output_dir='eda_outputs'):
    print("\nStarting Exploratory Data Analysis (EDA)...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Data
    df = pd.read_csv(filepath)

    # Open a text file to write the structured EDA report
    report_path = f"{output_dir}/1_EDA_Structured_Report.txt"
    with open(report_path, 'w') as f:
        f.write("="*40 + "\n")
        f.write("EXPLORATORY DATA ANALYSIS (EDA) REPORT\n")
        f.write("="*40 + "\n\n")

        # 2. Dataset Overview & Missing Values
        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Rows: {df.shape[0]}\n")
        f.write(f"Total Columns: {df.shape[1]}\n\n")
        
        f.write("Missing Values Count:\n")
        missing_data = df.isnull().sum()
        f.write(missing_data.to_string() + "\n\n")

        # 3. Statistical Summary
        f.write("2. STATISTICAL SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(df.describe().round(2).to_string() + "\n\n")

        # 4. Correlation Analysis
        f.write("3. CORRELATION INSIGHTS\n")
        f.write("-" * 20 + "\n")
        corr = df.drop('CustomerID', axis=1).corr()
        correlations = corr['Total_Spent'].sort_values(ascending=False).drop('Total_Spent')
        f.write("Key Factors Influencing 'Total_Spent':\n")
        f.write(correlations.round(3).to_string() + "\n\n")
        f.write("Insight: Variables with positive correlation close to 1 have a strong upward impact on Total_Spent.\n")

    print(f"Structured report saved to: {report_path}")

    # 5. Visualizations
    print("Generating statistical visualizations...")
    sns.set_theme(style="whitegrid")

    # Chart 1: Distribution Plot
    plt.figure(figsize=(9, 5))
    sns.histplot(df['Total_Spent'], bins=30, kde=True, color='indigo')
    plt.title('Distribution of Customer Spend (Total_Spent)')
    plt.xlabel('Total Spent ($)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/2_Spend_Distribution.png")
    plt.close()

    # Chart 2: Scatter Plot to show relationship/trends
    plt.figure(figsize=(9, 5))
    sns.scatterplot(x='Income', y='Total_Spent', data=df, alpha=0.5, color='darkcyan')
    plt.title('Trend Analysis: Income vs. Total Spent')
    plt.xlabel('Income ($)')
    plt.ylabel('Total Spent ($)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_Income_vs_Spend_Trend.png")
    plt.close()

    # Chart 3: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of Variables')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_Correlation_Heatmap.png")
    plt.close()

    print(f"Visualizations saved in the '{output_dir}' directory.")

if __name__ == "__main__":
    # Generate the dataset
    dataset_file = generate_mock_data()
    
    # Run the EDA pipeline
    perform_eda(dataset_file)
    print("\nEDA Project workflow completed successfully!")
