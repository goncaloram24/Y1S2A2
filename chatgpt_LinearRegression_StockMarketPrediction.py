import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data
df = pd.read_csv('stock_data.csv')

# Select the features and target
X = df[['interest_rate', 'unemployment_rate']]
y = df['stock_index_price']

# Create the model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Print the predictions
print(predictions)
