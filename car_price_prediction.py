import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import linear_regression as lgr
plt.style.use('ggplot')

car_df=pd.read_csv("car data.csv")
#general view of the dataset
print(car_df.head())
print(car_df.tail())
print(car_df.info())
print(car_df.describe())

#checking for duplicate rows
duplicate_rows = car_df[car_df.duplicated(keep=False)]
print("\n\nDuplicate rows:\n",duplicate_rows)
#dropping duplicate rows
car_df=car_df.drop_duplicates()

#checking for null values
print("\nNull values:\n",car_df.isnull().sum())

#one hot encoding data
car_df_encoded = pd.get_dummies(car_df, columns=['Fuel_Type','Selling_type','Transmission'])
print(car_df_encoded.head())

#visualization
#barcharts
#barchart to visualize use of fuel type
fig,axs=plt.subplots(figsize=(8, 5))
values = [
    len(car_df[car_df["Fuel_Type"] == "Petrol"]),
    len(car_df[car_df["Fuel_Type"] == "Diesel"]),
    len(car_df[(car_df["Fuel_Type"] != "Petrol") & (car_df["Fuel_Type"] != "Diesel")])
]
labels = ["Petrol", "Diesel", "Other"]
plt.bar(labels, values)
plt.xlabel("Fuel type")
plt.ylabel("count")
plt.title("Fuel types and their usage")

# grouped bar chart: Avg Selling Price vs Present Price over Years
# Calculate averages per year
grouped = car_df.groupby("Year")[["Selling_Price", "Present_Price"]].mean().sort_index()

years = grouped.index.astype(str)
selling_price_avg = grouped["Selling_Price"]
present_price_avg = grouped["Present_Price"]

# Setup positions
x = range(len(years))
w = 0.35  # bar width

# Plot grouped bars
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar([i - w/2 for i in x], selling_price_avg, width=w, label="Avg Selling Price", color="#6A5ACD")
ax.bar([i + w/2 for i in x], present_price_avg, width=w, label="Avg Present Price", color="#20B2AA")

ax.set_xticks(x)
ax.set_xticklabels(years, rotation=45)
ax.set_xlabel("Year")
ax.set_ylabel("Price (in lakhs)")
ax.set_title("Average Selling Price vs Present Price Over the Years")
ax.legend()


#piecharts
fig,axs1=plt.subplots(1,3)
#piechart 1
values = [
    len(car_df[car_df["Fuel_Type"] == "Petrol"]),
    len(car_df[car_df["Fuel_Type"] == "Diesel"]),
    len(car_df[(car_df["Fuel_Type"] != "Petrol") & (car_df["Fuel_Type"] != "Diesel")])
]
labels = ["Petrol", "Diesel", "Other"]
axs1[0].pie(values, labels=labels,autopct='%1.1f%%', startangle=90)
axs1[0].set_title("Fuel Type Distribution")

#piechart 2
values = [
    len(car_df[car_df["Selling_type"] == "Dealer"]),
    len(car_df[car_df["Selling_type"] == "Individual"])
]
labels = ["Dealer", "Individual"]
axs1[1].pie(values, labels=labels,autopct='%1.1f%%', startangle=90)
axs1[1].set_title("Selling Type Distribution")

#piechart 3
values = [
    len(car_df[car_df["Transmission"] == "Manual"]),
    len(car_df[car_df["Transmission"] == "Automatic"])
]
labels = ["Manual", "Automatic"]
axs1[2].pie(values, labels=labels,autopct='%1.1f%%', startangle=90)
axs1[2].set_title("Transmission Distribution")

#histogram
plt.figure(figsize=(12,6))
sns.histplot(data=car_df, x="Selling_Price", kde=True, label="Selling price")
sns.histplot(data=car_df, x="Present_Price", kde=True, label="Present price")
plt.title("Distribution of Selling vs Present Price")
plt.xlabel("Price (in lakhs)")
plt.ylabel("Count")
plt.legend()

#boxplots
plt.figure(figsize=(8,6))
plt.boxplot(car_df["Driven_kms"], tick_labels=["Driven"])
plt.ylabel("Kilometers")

#scatterplots
fig,axs2=plt.subplots(1,2)
axs2[0].scatter(car_df["Present_Price"], car_df["Selling_Price"])
axs2[1].scatter(car_df["Driven_kms"], car_df["Selling_Price"])

#correlation heatmap
corr = car_df.corr(numeric_only=True)

# Create the plot
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')  
plt.colorbar(label='Correlation Coefficient')

labels = corr.columns
plt.xticks(np.arange(len(labels)), labels, rotation=45, ha='right')
plt.yticks(np.arange(len(labels)), labels)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                 ha='center', va='center',
                 color='black' if abs(corr.iloc[i, j]) < 0.5 else 'white',
                 fontsize=9)

plt.title('Correlation Heatmap', fontsize=16)

plt.grid(False)

plt.tight_layout()
plt.show()


#linear regression from scratch
# Define feature matrix and target

X = car_df_encoded.drop(["Car_Name", "Selling_Price"], axis=1)
y = car_df_encoded["Selling_Price"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model and predict
w, b, history = lgr.gradient_descent(X_train, y_train.to_numpy(), num_iterations=800, learning_rate=0.05)
predictions = lgr.predict_all(X_test, w, b)

# Evaluate model
print("\nMy Model prediction metrics:\n")
r2 = r2_score(y_test, predictions)
print(f"R² Score: {r2:.4f} ({round(r2 * 100, 2)}%)")

mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error (MSE): {mse:.2f}")


#scikit's linear regression
pipeline= Pipeline([
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)  

print("\nScikit's Model prediction metrics:\n")
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f} ({round(r2 * 100, 2)}%)")

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")


# Compare Actual vs Predicted side-by-side
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# My Model
axs[0].scatter(y_test, predictions, alpha=0.6, color="mediumseagreen")
axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axs[0].set_title("My Linear Regression Model")
axs[0].set_xlabel("Actual Selling Price")
axs[0].set_ylabel("Predicted Selling Price")

# Scikit Model
axs[1].scatter(y_test, y_pred, alpha=0.6, color="cornflowerblue")
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axs[1].set_title("Scikit-Learn Linear Regression")
axs[1].set_xlabel("Actual Selling Price")
axs[1].set_ylabel("Predicted Selling Price")

plt.suptitle("Actual vs Predicted Selling Prices", fontsize=14)
plt.tight_layout()
plt.grid(True)
plt.show()

