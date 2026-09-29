# 🎓 Student Performance Prediction System

An end-to-end **Machine Learning regression system** that predicts a
student's **final exam score** using academic, behavioral, demographic,
and learning-related features.

The project uses **XGBoost Regression** with **RandomizedSearchCV** for
hyperparameter optimization, along with **Label Encoding**, **Standard
Scaling**, data-leakage prevention, model evaluation, and visualization.

> **Project Focus:** Predict final exam score → classify expected result
> as **PASS / FAIL** and identify the student's **performance level**.

------------------------------------------------------------------------

## 📌 Project Overview

Student academic performance can be influenced by multiple factors such
as:

-   Study hours
-   Attendance
-   Previous GPA
-   Tutoring sessions
-   Parent education
-   Internet access
-   Extracurricular activities
-   Sleep duration
-   Physical activity
-   School characteristics

This project builds a regression model that learns these relationships
and predicts a student's expected **final exam score (0--100)**.

The predicted score can then be converted into:

    Predicted Score Result   Performance
  ----------------- -------- -------------
             `< 40` FAIL     Poor
         `40–49.99` PASS     Average
         `50–69.99` PASS     Good
         `70–84.99` PASS     Very Good
            `>= 85` PASS     Excellent

------------------------------------------------------------------------

## 🎯 Objectives

-   Perform complete dataset exploration and preprocessing
-   Identify numerical and categorical features
-   Prevent target/data leakage
-   Encode categorical variables
-   Scale numerical variables
-   Build a baseline XGBoost regression model
-   Optimize XGBoost using `RandomizedSearchCV`
-   Compare baseline and optimized models
-   Evaluate the model using MAE, RMSE, and R²
-   Analyze feature importance
-   Visualize prediction quality and residual errors
-   Generate student-level prediction results
-   Convert predicted scores into PASS/FAIL and performance categories

------------------------------------------------------------------------

## 🗂️ Dataset

**Dataset size:** 5,000 students × 16 columns

### Features

  ------------------------------------------------------------------------------
  Feature                        Type                    Description
  ------------------------------ ----------------------- -----------------------
  `student_id`                   Identifier              Unique student
                                                         identifier

  `school_type`                  Categorical             Type of school

  `school_location`              Categorical             School location

  `gender`                       Categorical             Student gender

  `parent_education`             Categorical             Parent education level

  `internet_access`              Categorical             Internet availability

  `extracurricular_activities`   Categorical             Participation in
                                                         extracurricular
                                                         activities

  `tutoring_sessions`            Categorical             Tutoring-session
                                                         category

  `study_hours_per_week`         Numerical               Weekly study hours

  `attendance_rate_pct`          Numerical               Attendance percentage

  `previous_gpa`                 Numerical               Previous academic GPA

  `sleep_hours_per_night`        Numerical               Average nightly sleep

  `physical_activity_hours`      Numerical               Physical activity hours

  `final_exam_score`             Target                  Final exam score

  `pass_fail`                    Derived                 Pass/fail label derived
                                                         from final score

  `letter_grade`                 Derived                 Grade derived from
                                                         final score
  ------------------------------------------------------------------------------

### Data Quality

-   Missing values: **0**
-   Duplicate rows: **0**
-   Original rows: **5,000**
-   Target: `final_exam_score`

------------------------------------------------------------------------

## ⚠️ Data Leakage Prevention

A major part of this project is preventing information leakage.

The following columns are removed before training:

``` text
student_id
pass_fail
letter_grade
```

### Why?

`pass_fail` and `letter_grade` are derived from the final exam score.
Using them as input features would allow the model to indirectly see the
target.

`student_id` is only an identifier and does not provide meaningful
predictive information.

Therefore:

``` text
Raw Dataset
     ↓
Remove ID + Target-Derived Columns
     ↓
Feature Matrix
     ↓
Train/Test Split
     ↓
Encoding + Scaling
     ↓
Model Training
```

------------------------------------------------------------------------

## 🔄 Machine Learning Workflow

``` text
Dataset
   │
   ▼
Data Loading
   │
   ▼
Data Quality Check
   │
   ├── Missing Values
   ├── Duplicate Rows
   └── Statistical Summary
   │
   ▼
Exploratory Data Analysis
   │
   ├── Target Distribution
   ├── Boxplot
   ├── Categorical Distributions
   ├── Correlation Heatmap
   └── Feature vs Target Plots
   │
   ▼
Data Leakage Prevention
   │
   ▼
Feature / Target Separation
   │
   ▼
Train / Test Split
   │
   ├── Training: 80%
   └── Testing: 20%
   │
   ▼
Preprocessing
   │
   ├── LabelEncoder → Categorical Features
   └── StandardScaler → Numerical Features
   │
   ▼
Baseline XGBoost
   │
   ▼
RandomizedSearchCV
   │
   ├── 5-Fold Cross Validation
   └── 40 Random Hyperparameter Combinations
   │
   ▼
Optimized XGBoost
   │
   ▼
Final Prediction
   │
   ├── MAE
   ├── RMSE
   └── R²
   │
   ▼
Prediction Analysis
   │
   ├── Actual vs Predicted
   ├── Residual Analysis
   ├── Feature Importance
   └── Hyperparameter Comparison
   │
   ▼
Student Performance Classification
   │
   ├── PASS / FAIL
   └── Performance Level
```

------------------------------------------------------------------------

## 🧠 Preprocessing

### 1. Categorical Encoding

Categorical variables are converted into numerical representations using
`LabelEncoder`.

``` python
LabelEncoder()
```

Encoding is fitted only on the training data and then applied to the
test data.

### 2. Numerical Scaling

Numerical features are standardized using:

``` python
StandardScaler()
```

The scaler is fitted only on training data:

``` python
scaler.fit_transform(X_train)
```

and then applied to test data:

``` python
scaler.transform(X_test)
```

This prevents test-set information from influencing preprocessing.

------------------------------------------------------------------------

## 🤖 Model

### XGBoost Regressor

The primary model is:

``` python
XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
```

XGBoost was selected because it is a powerful gradient-boosting
algorithm that can capture non-linear relationships between student
characteristics and exam performance.

------------------------------------------------------------------------

## 🔧 Hyperparameter Optimization

Instead of relying only on default XGBoost parameters, the project uses:

``` python
RandomizedSearchCV
```

### Configuration

  Parameter                            Value
  --------------------- --------------------
  Search Method           RandomizedSearchCV
  Iterations                              40
  Cross Validation                    5-Fold
  Scoring                      Negative RMSE
  Random State                            42
  Parallel Processing            `n_jobs=-1`

### Search Space

The optimization searches across:

-   `n_estimators`
-   `max_depth`
-   `learning_rate`
-   `subsample`
-   `colsample_bytree`
-   `min_child_weight`
-   `gamma`
-   `reg_alpha`
-   `reg_lambda`

### Best Parameters Found

The current experiment produced:

``` text
n_estimators       = 300
max_depth          = 2
learning_rate      = 0.1
subsample          = 0.8
colsample_bytree   = 0.7
min_child_weight   = 7
gamma              = 0.3
reg_alpha          = 1
reg_lambda         = 5
```

Best 5-fold cross-validation RMSE:

``` text
4.0899
```

------------------------------------------------------------------------

## 📊 Model Performance

The optimized model was evaluated on a held-out 20% test set.

  Metric       Baseline XGBoost   Optimized XGBoost
  ---------- ------------------ -------------------
  MAE                    3.6868          **3.3067**
  RMSE                   4.5766          **4.1648**
  R² Score               0.8040          **0.8377**

### Metric Meaning

**MAE --- Mean Absolute Error**

Average absolute difference between actual and predicted scores.

``` text
Lower MAE → smaller average prediction error
```

**RMSE --- Root Mean Squared Error**

Penalizes larger prediction errors more strongly.

``` text
Lower RMSE → fewer large prediction errors
```

**R² Score**

Measures how much variance in the target is explained by the model.

``` text
Higher R² → better explanatory performance
```

------------------------------------------------------------------------

## 📈 Visualizations

The project includes multiple visual analyses:

### 1. Final Exam Score Distribution

Shows the distribution of student exam scores.

### 2. Target Boxplot

Helps inspect score spread and potential outliers.

### 3. Categorical Feature Distributions

Count plots are generated for categorical variables.

### 4. Correlation Heatmap

Shows relationships between numerical features and final exam score.

### 5. Feature vs Target Plots

Scatter plots analyze relationships such as:

``` text
Study Hours → Final Exam Score
Attendance → Final Exam Score
Previous GPA → Final Exam Score
Sleep Hours → Final Exam Score
Physical Activity → Final Exam Score
```

### 6. Baseline vs Optimized Model

Compares model performance across MAE, RMSE, and R².

### 7. Actual vs Predicted

Shows how closely predictions follow actual exam scores.

### 8. Residual Plot

Helps analyze prediction errors.

### 9. Residual Distribution

Shows the distribution of model errors.

### 10. Feature Importance

Displays which features contributed most to the XGBoost model's
predictions.

### 11. RandomizedSearchCV Results

Visualizes the top hyperparameter combinations based on cross-validation
RMSE.

------------------------------------------------------------------------

## 🎓 Student Prediction Logic

The final prediction is clipped to the valid exam-score range:

``` python
prediction = np.clip(prediction, 0, 100)
```

Then the predicted score is converted into an academic result.

Example:

``` text
Predicted Score: 78.5

Result:
PASS

Performance:
Very Good
```

Another example:

``` text
Predicted Score: 34.2

Result:
FAIL

Performance:
Poor
```

------------------------------------------------------------------------

## 🧪 Example Prediction Flow

``` text
Student Input
     ↓
Categorical Encoding
     ↓
Numerical Scaling
     ↓
Optimized XGBoost
     ↓
Predicted Exam Score
     ↓
Clip Score to 0–100
     ↓
PASS / FAIL
     ↓
Performance Level
```

------------------------------------------------------------------------

## 🛠️ Technologies Used

  Technology         Purpose
  ------------------ ----------------------------------------------
  Python             Core programming language
  Pandas             Data manipulation
  NumPy              Numerical computation
  Matplotlib         Visualization
  Seaborn            Statistical visualization
  Scikit-learn       Preprocessing, splitting, tuning, evaluation
  XGBoost            Regression model
  Jupyter Notebook   Development and experimentation

------------------------------------------------------------------------

## 📁 Project Structure

``` text
student-performance-prediction/
│
├── STUDENT_PERFORMANCE_PREDICTION_SYSTEM.ipynb
├── student_performance_prediction_dataset.csv
├── student_prediction_results.csv
└── README.md
```

------------------------------------------------------------------------

## ▶️ How to Run

### 1. Clone the repository

``` bash
git clone https://github.com/mdaasik-tech/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Install dependencies

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 3. Start Jupyter Notebook

``` bash
jupyter notebook
```

### 4. Open

``` text
STUDENT_PERFORMANCE_PREDICTION_SYSTEM.ipynb
```

### 5. Run all cells

The notebook performs:

``` text
Data Analysis
→ Preprocessing
→ Baseline Model
→ Hyperparameter Tuning
→ Final Model
→ Evaluation
→ Visualization
→ Prediction
```

------------------------------------------------------------------------

## 🔍 Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

-   Exploratory Data Analysis
-   Feature selection
-   Data leakage prevention
-   Train/test splitting
-   Label encoding
-   Standardization
-   Gradient boosting
-   XGBoost regression
-   Hyperparameter optimization
-   RandomizedSearchCV
-   Cross-validation
-   Regression evaluation
-   Residual analysis
-   Feature importance
-   Model comparison
-   Prediction post-processing
-   Rule-based performance classification

------------------------------------------------------------------------

## 🚀 Future Improvements

Possible next steps for production-level development:

-   Replace `LabelEncoder` with a production-safe categorical encoding
    pipeline
-   Use `Pipeline` / `ColumnTransformer` for cleaner preprocessing
-   Add model persistence using `joblib`
-   Build a FastAPI prediction API
-   Create a Streamlit dashboard
-   Add automated input validation
-   Add experiment tracking
-   Add MLflow for model/version tracking
-   Add Docker containerization
-   Add CI/CD
-   Deploy the model to a cloud platform
-   Add monitoring for prediction drift and data drift

------------------------------------------------------------------------

## ⚠️ Important Note

This project is intended for **educational and portfolio purposes**.

The model predicts exam performance from the features available in the
dataset. Real-world student performance can depend on many additional
factors that are not represented here.

Predictions should therefore be treated as model estimates rather than
guaranteed academic outcomes.

------------------------------------------------------------------------

## 👨‍💻 Author

### Mohammed Aasik

**Aspiring Machine Learning Engineer**

B.Sc. Artificial Intelligence & Machine Learning

-   GitHub: https://github.com/mdaasik-tech
-   LinkedIn:
    https://www.linkedin.com/in/mohammed-aasik-aspiring-machine-learning-engineer-257787433/

------------------------------------------------------------------------

## ⭐ If You Find This Project Useful

Feel free to explore the repository, review the notebook, and build on
the project.

**Machine Learning → Model Optimization → Deployment → MLOps**
