import streamlit as st
import pandas as pd

st.title("Customer purchasing behavior dataset analysis")
df=pd.read_csv("Customer-Purchasing-Behaviors.csv")

st.write("First 5 rows")
st.write(df.head())

st.write("Last 5 rows")
st.write(df.tail())

st.write("Dataset Shape")
st.write(df.shape)

st.write("Data columns")
st.write(df.columns)

st.write("Data Types")
st.write(df.dtypes)

st.write("Missing Values")
st.write(df.isnull().sum())

st.write("Descriptive Statistics for Numerical Features")
st.write(df.describe())

st.write("Descriptive Statistics for All Features")
st.write(df.describe(include='all'))

st.write("Unique Regions")
st.write(df['region'].unique())

st.write("Customers per Region")
st.write(df['region'].value_counts())

st.write("Region Statistics")
st.write(df.groupby('region').mean())

st.title("Customer Segmentation Analysis")
df["age_group"]=pd.cut(df["age"], bins=[0,30,50,100], labels=["Young","Middle-aged","Senior"])
age_group_stats=df.groupby("age_group")[["purchase_amount","loyalty_score"]].mean()
st.write("Age Group Statistics")
st.dataframe(age_group_stats)

st.title("Regional Sales Performance")
regional_sales = df.groupby("region")["purchase_amount"].sum()
st.write("Total Sales by Region")
st.dataframe(regional_sales)
st.success(f"Highest purchasing region:{regional_sales.idxmax()} with ${regional_sales.max():,.2f}")

st.title("Income vs Spending Correlation")
correlation = df["annual_income"].corr(df["purchase_amount"])
st.write(f"Correlation = {correlation:.3f}")
if correlation>0.7:
    st.success("Strong Positive Relationship")
elif correlation>0.3:
    st.info("Moderate Positive Relationship")
else:
    st.warning("Weak Relationship")
    
st.header("Loyalty vs Purchase Frequency")
high_loyalty_freq = df[df["loyalty_score"]>7.0]["purchase_frequency"].mean()
low_loyalty_freq = df[df["loyalty_score"]<5.0]["purchase_frequency"].mean()
st.write("4.Purchase Frequency Comparison")
st.write(f"High loyalty (>7.0): {high_loyalty_freq:.2f}")
st.write(f"Low loyalty (<5.0): {low_loyalty_freq:.2f}")

st.header("High-value Customer Identification")
high_value_customers = df[(df["annual_income"]>60000)& (df["purchase_amount"]>500)]
st.write(f"5. High-value customers count: {len(high_value_customers)}")
st.dataframe(high_value_customers[["user_id",'age','annual_income','purchase_amount']].head())

st.header("Repeat Purchase Analysis")
top_freq_customers= df.nlargest(10,'purchase_frequency')
avg_loyalty_score = top_freq_customers['loyalty_score'].mean()
st.write(f"6. Avg loyalty of top 10 frequent customers:{avg_loyalty_score:.2f}")

st.header("Regional Loyalty Comparison")
regional_loyalty = df.groupby('region')['loyalty_score'].mean()
st.write(f"7.Average Loyalty Score by Region:")
st.dataframe(regional_loyalty)
st.success(f"Most loyal region: {regional_loyalty.idxmax()} with{regional_loyalty.max():.2f}")

st.header("Age Group Spending Habits")
df['age_decade']= (df['age']//10)*10
age_decade_spending = df.groupby('age_decade')['purchase_amount'].mean()
st.write(f"8. Average Spending by Age Decade:")
st.dataframe(age_decade_spending)

st.header(" 9: Income Tier Analysis")
income_bins = [0,40000,60000,float("inf")]
income_labels = ["Low","Medium","High"]
df["income_tier"] = pd.cut(df["annual_income"],bins=income_bins,labels=income_labels)
income_counts = df["income_tier"].value_counts()
st.write("Customers in Each Income Tier")
st.dataframe(income_counts)

st.header(" 10: Purchase Efficiency")
df["spending_ratio"] = ( df["purchase_amount"] / df["annual_income"])
highest_ratio = df.loc[df["spending_ratio"].idxmax()]
st.write("Customer with Highest Spending Ratio")
st.write(f"User ID: {highest_ratio['user_id']}")
st.write(f"Ratio: {highest_ratio['spending_ratio']:.4f}")

st.header(" 11: Purchase Frequency by Region")
north_freq = df[df["region"] == "North"]["purchase_frequency"].mean()
south_freq = df[df["region"] == "South"]["purchase_frequency"].mean()
st.write(f"North: {north_freq:.2f}")
st.write(f"South: {south_freq:.2f}")
st.write(f"Difference: {abs(north_freq - south_freq):.2f}")

st.header(" 12: Loyalty Tier Classification")
df["loyalty_tier"] = pd.cut(df["loyalty_score"],bins=[0,5,7,10],labels=["Low","Medium","High"])
loyalty_counts = df["loyalty_tier"].value_counts()
st.write("Customers per Loyalty Tier")
st.dataframe(loyalty_counts)

st.header(" 13: Top Spenders by Region")
top_spenders = df.loc[df.groupby("region")["purchase_amount"].idxmax()]
st.write("Top Spenders in Each Region")
st.dataframe(top_spenders[["region","user_id","age","purchase_amount"]])

st.header(" 14: Income vs Loyalty Relationship")
high_loyalty_income = df[df["loyalty_score"] > 8]["annual_income"].mean()
overall_income = df["annual_income"].mean()
st.write(f"14. Income comparison (High Loyalty vs Overall):")
st.write(f"High Loyalty Income: {high_loyalty_income:,.2f}")
st.write(f"Overall Income: {overall_income:,.2f}")
st.write(f"Difference: {high_loyalty_income - overall_income:,.2f}")

st.header(" 15: Frequency vs Loyalty Correlation")
corr = df["purchase_frequency"].corr(df["loyalty_score"])
st.write(f"Correlation: {corr:.3f}")

st.header(" 16: Age and Purchase Frequency")
young_freq = df[df["age"] < 30]["purchase_frequency"].mean()
senior_freq = df[df["age"] > 50]["purchase_frequency"].mean()
st.write(f"Under 30: {young_freq:.2f}")
st.write(f"Over 50: {senior_freq:.2f}")
if young_freq > senior_freq:
    st.success("Young customers shop more frequently")
else:
    st.success("Senior customers shop more frequently")
    
st.header(" 17: Regional Income Comparison")
regional_income = df.groupby("region")["annual_income"].median()
st.dataframe(regional_income)
st.success(f"Highest Median Income Region: {regional_income.idxmax()} "
    f"({regional_income.max():,.2f})") 

st.header(" 18: Inconsistent Customers")
inconsistent = df[(df["purchase_frequency"] > 20) &(df["loyalty_score"] < 6)]
st.write(f"18. Number of inconsistent customers: {len(inconsistent)}")
st.dataframe(inconsistent[["user_id","purchase_frequency","loyalty_score"]].head())  

st.header(" 19: Spending per Visit")
df["spending_per_visit"] = df["purchase_amount"] / df["purchase_frequency"]
top_spenders_visit = df.nlargest(5, "spending_per_visit")
st.write(f"19. Top 5 customers by spending per visit:")
st.dataframe(top_spenders_visit[ ["user_id","spending_per_visit","purchase_amount","purchase_frequency"]])

st.header(" 20: Anomaly Detection")
anomalies = df[(df["loyalty_score"] > 8.5) &(df["purchase_frequency"] < 15)]
avg_income = anomalies["annual_income"].mean()
st.write(f"20. Number of anomalies: {len(anomalies)}")
st.write(f"Average Income: ${avg_income:,.2f}")
st.dataframe(anomalies[["user_id","loyalty_score","purchase_frequency","annual_income"]])