import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Load the dataset
data = pd.read_csv("train.csv")

# Select and clean the data
data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']].dropna()

# Define features and target
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Save the trained model
joblib.dump(model, "house_price_model.pkl")

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Predict custom input
custom_input = [[2000, 3, 2]]
predicted_price = model.predict(custom_input)
print(f"Predicted price for 2000 sqft, 3 bedrooms, 2 bathrooms: ₹{predicted_price[0]:,.2f}")

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_test['GrLivArea'], y_test, color='blue', label='Actual Prices', alpha=0.6)
plt.scatter(X_test['GrLivArea'], y_pred, color='red', label='Predicted Prices', alpha=0.6)
plt.xlabel('Living Area (GrLivArea)')
plt.ylabel('Sale Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

