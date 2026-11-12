"""Train and evaluate the student marks regression model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "students_marks.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = MODEL_DIR / "student_marks_model.pkl"
REPORT_PATH = OUTPUT_DIR / "evaluation.txt"

FEATURES = [
    "previous_marks",
    "attendance",
    "study_hours",
    "assignment_marks",
    "quiz_average",
]
TARGET = "final_marks"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if data[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing values. Clean the data before training.")

    print("Dataset loaded successfully!")
    print("Total records:", len(data))

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"\nTraining records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    print("\nTraining started...")
    model.fit(X_train, y_train)
    print("Training completed!")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions) if len(y_test) > 1 else float("nan")

    results = pd.DataFrame(
        {
            "Actual Marks": y_test.to_numpy(),
            "Predicted Marks": predictions.round(2),
        },
        index=y_test.index,
    ).sort_index()

    print("\n--- MODEL EVALUATION ---")
    print(f"MAE:  {mae:.2f}")
    print(f"MSE:  {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.2f}")

    print("\n--- PREDICTION COMPARISON ---")
    print(results.to_string(index=False))

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    report = (
        "STUDENT MARKS MODEL — EVALUATION REPORT\n"
        f"Dataset records: {len(data)}\n"
        f"Training records: {len(X_train)}\n"
        f"Testing records: {len(X_test)}\n"
        f"MAE: {mae:.2f}\n"
        f"MSE: {mse:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"R2: {r2:.2f}\n\n"
        "Actual vs predicted:\n"
        f"{results.to_string()}\n\n"
        "Caution: the included dataset is synthetic and very small. "
        "These metrics are educational examples, not evidence of real-world accuracy.\n"
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"\nModel saved to: {MODEL_PATH.relative_to(BASE_DIR)}")
    print(f"Evaluation report saved to: {REPORT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
