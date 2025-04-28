
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.autolayout': True})

# Load the dataset
file_path = r"C:\Users\USER\Downloads\realtime_data.csv"
df = pd.read_csv(file_path)

# -------------------------
# 1. DATA CLEANING & VISUALIZATION
# -------------------------

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Convert date
df['lastUpdated'] = pd.to_datetime(df['lastUpdated'], format='%d-%m-%Y', errors='coerce')

# Drop duplicates
df = df.drop_duplicates()

# Total beneficiaries by state
state_beneficiaries = df.groupby('state_name')['total_beneficiaries'].sum().reset_index().sort_values(by='total_beneficiaries', ascending=False)

# Bar plot - Total Beneficiaries by State
plt.figure(figsize=(14, 6))
#sns.barplot(data=state_beneficiaries, x='state_name', y='total_beneficiaries', palette='Spectral', dodge=False)
sns.barplot(data=state_beneficiaries, x='state_name', y='total_beneficiaries', hue='state_name', palette='Spectral', dodge=False, legend=False)

plt.xticks(rotation=90)
plt.title("Total Beneficiaries by State")
plt.xlabel("State")
plt.ylabel("Total Beneficiaries")
plt.tight_layout()
plt.show()

# -------------------------
# 2. EDA & STATISTICAL ANALYSIS
# -------------------------

#  Correlation Matrix Heatmap
plt.figure(figsize=(10, 7))
correlation = df[['total_beneficiaries', 'total_aadhar', 'total_mobileno', 'total_sc', 'total_st', 'total_gen', 'total_obc']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Scatterplot - Aadhaar vs Beneficiaries
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='total_aadhar', y='total_beneficiaries', hue='state_name', palette='tab20', alpha=0.7)
plt.title("Aadhaar Linked vs Total Beneficiaries")
plt.xlabel("Aadhaar Linked")
plt.ylabel("Total Beneficiaries")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
plt.tight_layout()
plt.show()

# Correlation between aadhar and beneficiaries
cor_val = df['total_aadhar'].corr(df['total_beneficiaries'])
print(f"\nCorrelation between Total Aadhar and Total Beneficiaries: {cor_val}")

# -------------------------
# 3. CREATIVITY & INNOVATION
# -------------------------

# Inclusion Score = average of Aadhaar + Mobile coverage
df['inclusion_score'] = (df['total_aadhar'] + df['total_mobileno']) / (2 * df['total_beneficiaries'])
df['inclusion_score'] = df['inclusion_score'].clip(upper=1.0)

# Top 5 inclusive districts
top_districts = df.groupby('district_name')['inclusion_score'].mean().reset_index().sort_values(by='inclusion_score', ascending=False).head(5)
print("\nTop 5 Districts by Inclusion Score:\n", top_districts)

# Top 5 inclusive states
top_states = df.groupby('state_name')['inclusion_score'].mean().reset_index().sort_values(by='inclusion_score', ascending=False).head(5)
print("\nTop 5 States by Inclusion Score:\n", top_states)

# Plot top 5 states by Inclusion Score
'''plt.figure(figsize=(10, 5))
sns.barplot(data=top_states, x='state_name', y='inclusion_score', palette='coolwarm', dodge=False)
plt.title("Top 5 States by Inclusion Score")
plt.ylabel("Inclusion Score")
plt.xlabel("State")
plt.tight_layout()
plt.show()'''



# Trend in scheme reporting over time
monthly_report = df.copy()
monthly_report['month'] = monthly_report['lastUpdated'].dt.to_period('M')
monthly_trend = monthly_report.groupby('month').size().reset_index(name='report_count')

plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_trend, x='month', y='report_count', marker='o', color='purple')
plt.title("Monthly Scheme Data Reporting Trend")
plt.xlabel("Month")
plt.ylabel("Reports Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# District-wise beneficiary analysis (Top 20)
top_districts_beneficiaries = df.groupby('district_name')['total_beneficiaries'].sum().reset_index().sort_values(by='total_beneficiaries', ascending=False).head(20)

'''plt.figure(figsize=(12, 6))
sns.barplot(data=top_districts_beneficiaries, x='district_name', y='total_beneficiaries', palette='viridis')
plt.title("Top 20 Districts by Total Beneficiaries")
plt.xlabel("District")
plt.ylabel("Total Beneficiaries")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()'''

plt.figure(figsize=(12, 6))
sns.barplot(data=top_districts_beneficiaries, x='district_name', y='total_beneficiaries', hue='district_name', palette='viridis', dodge=False, legend=False)
plt.title("Top 20 Districts by Total Beneficiaries")
plt.xlabel("District")
plt.ylabel("Total Beneficiaries")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# -------------------------
# Optional: Save static plot as PDF
# -------------------------

plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['total_beneficiaries', 'total_aadhar', 'total_mobileno']], palette='Set2')
plt.title("Boxplot for Key Metrics")
plt.tight_layout()
#plt.savefig("key_metrics_boxplot.pdf")
plt.show()
