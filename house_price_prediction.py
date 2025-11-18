# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# STEP 2: Load Dataset
df = pd.read_csv('/content/train.csv')
df.head()

# STEP 3: Select the Important Features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
df = df[features + ['SalePrice']]

# STEP 4: Handle Missing Values
df = df.dropna()

# STEP 5: Graph (Living Area vs Sale Price)
plt.scatter(df['GrLivArea'], df['SalePrice'])
plt.xlabel("Living Area (sq ft)")
plt.ylabel("Sale Price")
plt.title("House Price vs Living Area")
plt.show()

# STEP 6: Train-Test Split
X = df[features]          # GrLivArea, BedroomAbvGr, FullBath
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 7: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 8: Evaluate Model
y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# STEP 9: Predict Price for a New House
new_house = pd.DataFrame({
    'GrLivArea': [2000],     # square feet
    'BedroomAbvGr': [3],     # bedrooms
    'FullBath': [2]          # bathrooms
})

predicted_price = model.predict(new_house)
print("Predicted Price for the New House:", predicted_price[0])
