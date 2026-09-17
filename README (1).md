# Car Price Regression with Missing Data

This project demonstrates a small tabular machine-learning workflow for predicting car prices using Scikit-Learn.

The project was developed from my original coursework and cleaned up for portfolio presentation while preserving the original modeling approach.

## Dataset

The dataset contains 1,000 rows with these columns:

- `Make`
- `Colour`
- `Odometer (KM)`
- `Doors`
- `Price`

The version used in this project includes missing values, which makes the preprocessing workflow more realistic.

## Workflow

The notebook includes:

- loading and inspecting the dataset
- checking missing values
- removing rows with missing target values
- separating features and target
- imputing missing categorical and numerical values
- converting categorical features with `OneHotEncoder`
- using `ColumnTransformer`
- splitting data into training and test sets
- training a `RandomForestRegressor`
- evaluating the model with R-squared

## Model

The final model uses:

- `RandomForestRegressor`
- `n_estimators=100`
- an 80/20 train/test split
- a fixed NumPy random seed for reproducibility

## Result

The saved run produces a test-set R-squared of approximately:

**0.22**

This is not a strong predictive result, so the project should be viewed as a learning-focused demonstration of tabular preprocessing and regression rather than a production-ready car-price model.

## Limitations

- The dataset contains only 1,000 rows.
- The available features are limited and omit important pricing factors such as model year, trim, condition, accident history, and location.
- The final R-squared is modest.
- The original coursework preprocessing order fits the imputer and encoder before the train/test split. A stricter workflow would split the data first and fit preprocessing only on the training data to reduce information leakage.
- No additional models, hyperparameter tuning, pipelines, or new modeling strategies were added for this portfolio version.

## Files

- `car_price_regression.ipynb` — cleaned portfolio notebook
- `car-sales-extended-missing-data.csv` — dataset used by the notebook

## Libraries Used

- pandas
- NumPy
- Scikit-Learn
