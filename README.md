# Titanic Survival Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster, using the classic [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic) and a Logistic Regression classifier built with scikit-learn.

## Project structure

```
.
├── titanic.ipynb            # Full notebook: data cleaning, EDA, model training & evaluation
├── train.csv                # Training data (891 passengers, with Survived labels)
├── test.csv                 # Kaggle test data (no labels, for submission)
├── gender_submission.csv    # Kaggle's sample submission format
└── README.md
```

## Dataset overview

The training set (`train.csv`) contains **891 passengers** and **12 columns**:

| Column | Description |
|---|---|
| PassengerId | Row identifier |
| Survived | Target: 0 = did not survive, 1 = survived |
| Pclass | Ticket class (1st, 2nd, 3rd) |
| Name | Passenger name |
| Sex | male / female |
| Age | Age in years |
| SibSp | # of siblings/spouses aboard |
| Parch | # of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation (S, C, Q) |

**Missing values found:**
- `Age`: 177 missing (out of 891)
- `Cabin`: 687 missing (out of 891) — mostly empty, dropped entirely
- `Embarked`: 2 missing

**Class balance:**
- Did not survive (0): 549 (61.6%)
- Survived (1): 342 (38.4%)

**Sex distribution:** 577 male, 314 female
**Embarked distribution:** S: 646, C: 168, Q: 77

## Methodology

1. **Data cleaning**
   - Dropped the `Cabin` column (77% missing, too sparse to be useful).
   - Filled missing `Age` values with the column mean.
   - Filled the 2 missing `Embarked` values with the column mode (`S`).

2. **Exploratory data analysis**
   - Summary statistics (`describe()`), survival counts, and count plots for `Survived`, `Sex`, `Pclass`, and `Embarked`, including survival breakdowns by sex and class.

3. **Feature engineering**
   - Encoded categorical columns to numeric: `Sex` (male=0, female=1) and `Embarked` (S=0, C=1, Q=2).
   - Dropped `PassengerId`, `Name`, and `Ticket` from the feature set — they carry no direct predictive signal in raw form.
   - Final feature set (`X`): `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked` (7 features).
   - Target (`Y`): `Survived`.

4. **Train/test split**
   - 80/20 split via `train_test_split(X, Y, test_size=0.2, random_state=2)`.
   - Training set: 712 passengers — Test set: 179 passengers.

5. **Model**
   - `sklearn.linear_model.LogisticRegression()`, trained with `model.fit(X_train, Y_train)`.

## Results

| Metric | Score |
|---|---|
| **Training accuracy** | 80.76% |
| **Test accuracy** | 78.21% |

The close gap between training and test accuracy (~2.5 points) indicates the model generalizes reasonably well and is not significantly overfitting.

> **Note:** The solver (`lbfgs`) hits its default iteration limit (100) without fully converging, which triggers a `ConvergenceWarning`. This doesn't invalidate the results, but accuracy could likely be improved slightly by increasing `max_iter` and/or scaling the numeric features (e.g. `Age`, `Fare`) before training.

## Tech stack

- Python
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — train/test split, Logistic Regression, accuracy scoring

## How to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook titanic.ipynb
```

Run all cells top to bottom — the notebook loads `train.csv` from the same directory, cleans the data, trains the model, and prints the accuracy scores.
