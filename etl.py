# walmart_analysis_code.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ======================== Load Dataset ========================
df = pd.read_csv("Walmart_customer_purchases.csv", parse_dates=["Purchase_Date"])

# ======================== Data Cleaning ========================
# Drop rows with missing values
df.dropna(inplace=True)

# ======================== Feature Engineering ========================

# 1. Repeat vs Non-repeat sales
repeat_sales = df.groupby("Repeat_Customer")["Purchase_Amount"].sum().reset_index()
repeat_sales.columns = ["repeat_customer", "total_sales"]

# 2. Gender-wise sales
gender_purchase = df.groupby("Gender")["Purchase_Amount"].sum().reset_index()
gender_purchase.columns = ["gender", "total_sales"]

# 3. Average ratings by category
category_ratings = df.groupby("Category")["Rating"].mean().reset_index()
category_ratings.columns = ["category", "average_rating"]

# 4. Discount impact on sales
discount_impact = df.groupby("Discount_Applied")["Purchase_Amount"].sum().reset_index()
discount_impact.columns = ["discount_applied", "total_sales"]

# 5. Age group based sales
bins = [0, 18, 30, 45, 60, 100]
labels = ["<18", "18-30", "30-45", "45-60", "60+"]
df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)
age_group_sales = df.groupby("age_group")["Purchase_Amount"].sum().reset_index()
age_group_sales.columns = ["age_group", "total_sales"]

# 6. Top products by category
top_n_per_cat = df.groupby(["Category", "Product_Name"])["Purchase_Amount"].sum().reset_index()
top_n_per_cat = top_n_per_cat.sort_values(["Category", "Purchase_Amount"], ascending=[True, False])
top_n_per_cat = top_n_per_cat.groupby("Category").head(3)

# 7. City-wise revenue
city_revenue = df.groupby("City")["Purchase_Amount"].sum().reset_index()

# 8. Sales by weekday
df["weekday"] = df["Purchase_Date"].dt.day_name()
weekday_sales = df.groupby("weekday")["Purchase_Amount"].sum().reset_index()

# 9. Sales by day of the month
df["day_of_month"] = df["Purchase_Date"].dt.day
daily_sales = df.groupby("day_of_month")["Purchase_Amount"].sum().reset_index()
daily_sales.columns = ["day_of_month", "total_sales"]

# ======================== Save All Outputs to CSV ========================
output_dir = "walmart_sales_analysis"
os.makedirs(output_dir, exist_ok=True)

df.to_csv(f"{output_dir}/cleaned_data.csv", index=False)
repeat_sales.to_csv(f"{output_dir}/repeat_vs_nonrepeat_sales.csv", index=False)
gender_purchase.to_csv(f"{output_dir}/gender_sales.csv", index=False)
category_ratings.to_csv(f"{output_dir}/category_ratings.csv", index=False)
discount_impact.to_csv(f"{output_dir}/discount_impact.csv", index=False)
age_group_sales.to_csv(f"{output_dir}/age_group_sales.csv", index=False)
top_n_per_cat.to_csv(f"{output_dir}/top_products_by_category.csv", index=False)
city_revenue.to_csv(f"{output_dir}/city_revenue.csv", index=False)
weekday_sales.to_csv(f"{output_dir}/weekday_sales.csv", index=False)
daily_sales.to_csv(f"{output_dir}/daily_sales_by_date_of_month.csv", index=False)

# ======================== Optional: Generate Plots ========================
# Enable inline plotting if in Jupyter or Colab
import matplotlib.pyplot as plt

# Plot: Repeat vs Non-repeat
repeat_sales.plot(kind="bar", x="repeat_customer", y="total_sales", title="Repeat vs Non-Repeat Sales")
plt.tight_layout()
plt.savefig(f"{output_dir}/plot_repeat_vs_nonrepeat.png")

# Plot: Gender-wise sales
gender_purchase.plot(kind="bar", x="gender", y="total_sales", title="Gender-wise Sales", color="orange")
plt.tight_layout()
plt.savefig(f"{output_dir}/plot_gender_sales.png")

# Plot: Discount impact
discount_impact.plot(kind="bar", x="discount_applied", y="total_sales", title="Discount Impact", color="green")
plt.tight_layout()
plt.savefig(f"{output_dir}/plot_discount_impact.png")

# Plot: Age group sales
age_group_sales.plot(kind="bar", x="age_group", y="total_sales", title="Age Group Sales", color="purple")
plt.tight_layout()
plt.savefig(f"{output_dir}/plot_age_group_sales.png")

# Plot: Sales by weekday
weekday_sales.plot(kind="bar", x="weekday", y="Purchase_Amount", title="Sales by Weekday", color="skyblue")
plt.tight_layout()
plt.savefig(f"{output_dir}/plot_weekday_sales.png")

# Plot: Sales by day of month
daily_sales.plot(kind="line", x="day_of_month", y="total_sales", title="Sales by Day of Month")
plt.tight_layout()
plt.savefig(f"{output_dir}/plot_daily_sales.png")

print("✅ All analysis complete. Files saved in:", output_dir)
