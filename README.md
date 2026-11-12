# Student Marks Prediction using Machine Learning

A beginner-friendly supervised machine learning project that predicts a student's final marks using previous academic performance, attendance, study hours, assignment marks, and quiz average.

This project uses **Python, Pandas, and Scikit-learn**, with a `RandomForestRegressor` model.

> **Important:** The included dataset contains only 20 synthetic records created for learning. The evaluation scores are for demonstration and must not be interpreted as evidence of real-world predictive performance.

## Project overview

- **Problem type:** Supervised learning — regression
- **Target:** `final_marks`
- **Input features:** `previous_marks`, `attendance`, `study_hours`, `assignment_marks`, `quiz_average`
- **Algorithm:** Random Forest Regressor
- **Train/test split:** 80% / 20%
- **Model persistence:** Joblib

## Project structure

```text
student-marks-ml/
├── data/
│   └── students_marks.csv
├── models/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── .gitignore
├── predict.py
├── requirements.txt
├── train_model.py
└── README.md
```

The trained model (`models/student_marks_model.pkl`) and evaluation report (`outputs/evaluation.txt`) are generated when you run the training script. They are intentionally excluded from Git by `.gitignore`.

## Requirements

- Python 3.10 or newer
- pip

## Setup on Windows PowerShell

Open PowerShell in the project directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run this once for your current Windows user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again.

## Train the model

```powershell
python train_model.py
```

The script:
1. Loads the CSV dataset.
2. Separates input features (`X`) from the target (`y`).
3. Splits the data into training and testing subsets.
4. Trains a Random Forest regression model.
5. Calculates MAE, MSE, RMSE, and R².
6. Prints actual-versus-predicted values.
7. Saves the model to `models/student_marks_model.pkl`.
8. Writes a report to `outputs/evaluation.txt`.

## Make a prediction

After training has completed:

```powershell
python predict.py
```

To predict for a different student, edit the values in the `student` dictionary inside `predict.py`. Keep values within sensible ranges (marks and attendance from 0 to 100; study hours greater than or equal to zero).

## Evaluation metrics

The original experiment on the included 20-row dataset produced:

| Metric | Original run |
|---|---:|
| MAE | 1.69 |
| MSE | 5.85 |
| RMSE | 2.42 |
| R² | 0.94 |

The exact results can vary with library versions or code changes. Only four records were used for testing, so these metrics are highly unstable. A high R² on this tiny synthetic dataset does not demonstrate that the model will generalize to real students.

## Dataset

`data/students_marks.csv` is a small synthetic teaching dataset. It does not contain real student information and is not suitable for academic decisions.

| Feature | Description |
|---|---|
| `previous_marks` | Previous examination marks |
| `attendance` | Attendance percentage |
| `study_hours` | Average daily study hours |
| `assignment_marks` | Assignment marks |
| `quiz_average` | Average quiz marks |
| `final_marks` | Final examination marks (prediction target) |

## Next improvements

- Collect a larger, representative, ethically sourced dataset.
- Use cross-validation rather than relying on a single four-row test set.
- Compare Linear Regression, Decision Tree, and Random Forest.
- Add feature-importance analysis and residual plots.
- Track experiments and document data provenance.
- Expose the model through FastAPI, then integrate it with a Spring Boot application.

## Portfolio note

This is an introductory ML experiment demonstrating the end-to-end workflow: data loading, feature/target separation, train/test splitting, model fitting, regression evaluation, model serialization, and inference.

## License

This project is provided for educational use. Add a license file if you want to specify reuse permissions for your repository.
