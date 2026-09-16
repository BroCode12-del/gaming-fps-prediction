import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
def evaluate_model(y_test, xgb_pred):
    mae = mean_absolute_error(y_test, xgb_pred)
    mse = mean_squared_error(y_test, xgb_pred)
    r2 = r2_score(y_test, xgb_pred)
    mape = mean_absolute_percentage_error(y_test, xgb_pred)

    print("MAE:", mae)
    print("MSE:", mse)
    print("R2:", r2)
    print("MAPE (%):", mape * 100)

    plt.figure()
    plt.title('XGBoost: Predicted vs Actual FPS')
    plt.scatter(xgb_pred, y_test, color='green')

    min_xgb = min(min(xgb_pred), min(y_test))
    max_xgb = max(max(xgb_pred), max(y_test))

    plt.plot(
        [min_xgb, max_xgb],
        [min_xgb, max_xgb],
        color='red',
        linestyle='--',
        lw=2,
        label='Perfect prediction'
    )

    plt.xlabel('Predicted FPS')
    plt.ylabel('True FPS')
    plt.legend()
    plt.tight_layout()
    plt.show()