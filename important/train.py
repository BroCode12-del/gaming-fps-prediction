import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from cleaning import clean_data
from preprocessing import prepare_features
from evaluation import evaluate_model
from pathlib import Path
import joblib

def main():
    df = clean_data()

    X, y, preprocessor = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # Keep original FPS values and use a separate log target.
    y_train_log = np.log1p(y_train)

    xgb_model = XGBRegressor(
        n_estimators=165,
        max_depth=4,
        learning_rate=0.07,
        objective='reg:squarederror',
        min_child_weight=3.3,
        subsample=0.6,
        reg_lambda=2
    )

    xgb_model.fit(X_train_encoded, y_train_log)

    # Convert predictions back to FPS before evaluation.
    xgb_pred = np.expm1(xgb_model.predict(X_test_encoded))

    model_folder = Path(__file__).resolve().parent / 'models'
    model_folder.mkdir(exist_ok=True)

    joblib.dump(
        {
            'model': xgb_model,
            'preprocessor': preprocessor,
            'feature_columns': X.columns.tolist(),
            'data': df,
            'y_test': y_test.to_numpy(),
            'predictions': xgb_pred
        },
        model_folder / 'fps_model.joblib'
    )

    print("Model saved to models/fps_model.joblib")

    evaluate_model(y_test, xgb_pred)


if __name__ == '__main__':
    main()