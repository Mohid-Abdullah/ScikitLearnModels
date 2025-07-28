import seaborn as sns
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_percentage_error,classification_report,confusion_matrix
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


df = sns.load_dataset('titanic')
df = df.dropna(subset=['age', 'embarked', 'fare'])
df = df[['survived', 'sex', 'age', 'fare', 'class', 'embarked']]

x = df.drop(columns="survived")
x_classs = df['class']
x_str = x.select_dtypes(include="object").columns.to_list()
x_values = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
y = df['survived']

preprocessor = ColumnTransformer(
    transformers=[
        ("oneHot", OneHotEncoder(handle_unknown='ignore'), x_str),
        ("oridinal", OrdinalEncoder(handle_unknown='error'), ['class']),
        ('number', StandardScaler(), x_values)
    ]
)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44, stratify=y)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42,class_weight='balanced'))
])

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print("\nClassificationReport:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

