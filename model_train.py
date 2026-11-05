import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('laptop_dataset.csv')
df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)

X = df.drop('price', axis=1)
y = df['price']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Pipeline for numeric features
num_transform = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

# Pipeline for categorical features
cat_transform = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Typo fixed
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ]
)

# Combine them using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transform, numeric_features),
        ('cat', cat_transform, categorical_features)
    ]
)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the full pipeline with preprocessor and Linear Regression model
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Fit the pipeline on training data
lr_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = lr_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Save the fitted pipeline
with open("laptop_pipeline.pkl", "wb") as f:
    pickle.dump(lr_pipeline, f)

print("Linear regressor pipeline saved as laptop_pipeline.pkl")