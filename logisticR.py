import kagglehub
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_percentage_error,classification_report,confusion_matrix
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# Download latest version
path = kagglehub.dataset_download("jessemostipak/hotel-booking-demand")

df = pd.read_csv(path)
print(df.count())
print(df.isnull().sum())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.drop(columns=["agent", "company"], inplace=True)

print(df.isnull().sum())

print(df.dtypes)

y = df['is_canceled']
x = df.drop(columns=["is_canceled"], axis=1)

x_objs = x.select_dtypes(include="object").columns.to_list()
x_others = x.select_dtypes(include=["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[("convert", OneHotEncoder(handle_unknown='ignore'), x_objs),
    ('number', StandardScaler(), x_others)]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000,class_weight='balanced'))
])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=44,stratify=y)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print("\nClassificationReport:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))




