"""Load the trained model and predict marks for one student."""

from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_marks_model.pkl"

FEATURES = [
    "previous_marks",
    "attendance",
    "study_hours",
    "assignment_marks",
    "quiz_average",
]


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run 'python train_model.py' first."
        )

    model = joblib.load(MODEL_PATH)

    # Change these sample values to predict for another student.
    student = {
        "previous_marks": 67,
        "attendance": 84,
        "study_hours": 3,
        "assignment_marks": 76,
        "quiz_average": 71,
    }

    if not 0 <= student["previous_marks"] <= 100:
        raise ValueError("previous_marks must be between 0 and 100.")
    if not 0 <= student["attendance"] <= 100:
        raise ValueError("attendance must be between 0 and 100.")
    if student["study_hours"] < 0:
        raise ValueError("study_hours cannot be negative.")
    for field in ("assignment_marks", "quiz_average"):
        if not 0 <= student[field] <= 100:
            raise ValueError(f"{field} must be between 0 and 100.")

    student_data = pd.DataFrame([student], columns=FEATURES)
    predicted_marks = model.predict(student_data)[0]

    print("Student Marks Prediction System")
    print("\nStudent Information:")
    print(student_data.to_string(index=False))
    print(f"\nPredicted Future Marks: {predicted_marks:.2f} / 100")
    print(
        "\nNote: This is an educational prediction from a small synthetic "
        "dataset, not a reliable assessment of a real student's performance."
    )


if __name__ == "__main__":
    main()
