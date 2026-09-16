import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

from cleaning import clean_data
from preprocessing import prepare_features


df = clean_data()

X, y, preprocessor = prepare_features(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train = np.log1p(y_train)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)


# Linear regression
linear_model = LinearRegression()
linear_model.fit(X_train_encoded, y_train)
linear_pred = np.expm1(linear_model.predict(X_test_encoded))

plt.figure()
plt.title('Linear Regression: Predicted vs Actual FPS')
plt.scatter(linear_pred, y_test, color='green')

min_lin = min(min(linear_pred), min(y_test))
max_lin = max(max(linear_pred), max(y_test))

plt.plot(
    [min_lin, max_lin],
    [min_lin, max_lin],
    color='red',
    linestyle='--',
    lw=2,
    label='Perfect prediction'
)

plt.xlabel('Predicted FPS')
plt.ylabel('True FPS')
plt.legend()
plt.show()

mae_lin = mean_absolute_error(y_test, linear_pred)
mse_lin = mean_squared_error(y_test, linear_pred)
r2_lin = r2_score(y_test, linear_pred)
mape_lin = mean_absolute_percentage_error(y_test, linear_pred)

print(
    f"Linear MAE, MSE, R2, MAPE (%): "
    f"{mae_lin}, {mse_lin}, {r2_lin}, {mape_lin * 100}"
)


# Polynomial regression
poly_model = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression()
)

poly_model.fit(X_train_encoded, y_train)
poly_pred = np.expm1(poly_model.predict(X_test_encoded))

plt.figure()
plt.title('Polynomial Regression: Predicted vs Actual FPS')
plt.scatter(poly_pred, y_test, color='green')

min_poly = min(min(poly_pred), min(y_test))
max_poly = max(max(poly_pred), max(y_test))

plt.plot(
    [min_poly, max_poly],
    [min_poly, max_poly],
    color='red',
    linestyle='--',
    lw=2,
    label='Perfect prediction'
)

plt.xlabel('Predicted FPS')
plt.ylabel('True FPS')
plt.legend()
plt.show()

mae_poly = mean_absolute_error(y_test, poly_pred)
mse_poly = mean_squared_error(y_test, poly_pred)
r2_poly = r2_score(y_test, poly_pred)
mape_poly = mean_absolute_percentage_error(y_test, poly_pred)

print(
    f"Poly MAE, MSE, R2, MAPE (%): "
    f"{mae_poly}, {mse_poly}, {r2_poly}, {mape_poly * 100}"
)


# Random forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train_encoded, y_train)
rf_pred = np.expm1(rf_model.predict(X_test_encoded))

plt.figure()
plt.title('Random Forest: Predicted vs Actual FPS')
plt.scatter(rf_pred, y_test, color='green')

min_rf = min(min(rf_pred), min(y_test))
max_rf = max(max(rf_pred), max(y_test))

plt.plot(
    [min_rf, max_rf],
    [min_rf, max_rf],
    color='red',
    linestyle='--',
    lw=2,
    label='Perfect prediction'
)

plt.xlabel('Predicted FPS')
plt.ylabel('True FPS')
plt.legend()
plt.show()

mae_rf = mean_absolute_error(y_test, rf_pred)
mse_rf = mean_squared_error(y_test, rf_pred)
r2_rf = r2_score(y_test, rf_pred)
mape_rf = mean_absolute_percentage_error(y_test, rf_pred)

print(
    f"Random Forest MAE, MSE, R2, MAPE (%): "
    f"{mae_rf}, {mse_rf}, {r2_rf}, {mape_rf * 100}"
)