import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


# Create required folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# Load dataset
DATA_PATH = "data/student_performance.csv"

df = pd.read_csv(DATA_PATH)

print("\nDATASET LOADED")
print("-------------------------")
print(df.head())
print("\nDataset shape:", df.shape)


# Features and target
features = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
    "sleep_hours",
    "extracurricular_hours"
]

X = df[features]
y = df["final_score"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Models
models = {

    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=5
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}


# 5-Fold Cross Validation
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []


# Train and evaluate models
for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="r2"
    )

    cv_r2 = cv_scores.mean()

    results.append([
        name,
        mae,
        rmse,
        r2,
        cv_r2
    ])

    filename = name.lower().replace(" ", "_") + ".joblib"

    joblib.dump(
        model,
        "models/" + filename
    )


# GridSearchCV
print("\nRunning GridSearchCV...")

parameters = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    parameters,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBEST PARAMETERS")
print(grid.best_params_)


# Tuned model evaluation
predictions = best_model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

tuned_cv_r2 = np.mean(
    cross_val_score(
        best_model,
        X,
        y,
        cv=kf,
        scoring="r2"
    )
)

results.append([
    "Tuned Random Forest",
    mae,
    rmse,
    r2,
    tuned_cv_r2
])


# Save tuned model
joblib.dump(
    best_model,
    "models/tuned_random_forest.joblib"
)


# Comparison table
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2",
        "5-Fold CV R2"
    ]
)

print("\nMODEL COMPARISON")
print("-------------------------")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# Feature importance
importance = pd.Series(
    best_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFEATURE IMPORTANCE")
print("-------------------------")
print(importance)

importance.to_csv(
    "outputs/feature_importance.csv"
)


# Feature importance graph
plt.figure(figsize=(9, 5))

importance.sort_values().plot(
    kind="barh"
)

plt.title(
    "Feature Importance - Random Forest"
)

plt.xlabel(
    "Importance"
)

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png"
)

plt.close()


# Final recommendation report
report = f"""
STUDENT PERFORMANCE PREDICTION MODEL
====================================

Models trained:
1. Linear Regression
2. Decision Tree
3. Random Forest
4. Tuned Random Forest using GridSearchCV

Cross-validation:
5-Fold Cross Validation

Evaluation metrics:
MAE
RMSE
R2

Best GridSearch parameters:
{grid.best_params_}

Model comparison:

{results_df.to_string(index=False)}

Feature importance:

{importance.to_string()}

Final model:
Tuned Random Forest

The project predicts a student's final academic
performance using study hours, attendance,
previous score, assignment score, sleep hours,
and extracurricular activity.
"""

with open(
    "outputs/final_recommendation_report.txt",
    "w"
) as file:
    file.write(report)


print("\n================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("================================")

print("\nGenerated files:")
print("outputs/model_comparison.csv")
print("outputs/feature_importance.csv")
print("outputs/feature_importance.png")
print("outputs/final_recommendation_report.txt")
print("models/tuned_random_forest.joblib")