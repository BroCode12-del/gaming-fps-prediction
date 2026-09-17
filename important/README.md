A Gaming FPS Prediction Project

A machine learning project that estimates average gaming FPS using CPU and GPU specifications, game information, and graphics settings.

I built this project to practice combining datasets, preprocessing mixed data types, comparing regression models, and evaluating predictions

 Project overview:

The project combines three CSV datasets:

- CPU specifications
- GPU specifications
- Gaming FPS records

Hardware records are matched to FPS records using `cpu_key` and `gpu_key`. The merged data includes features such as CPU cores and clock speeds, GPU memory and processing specifications, game, resolution, graphics preset, ray tracing, and upscaling settings.

The prediction target is `avg_fps`.

## Data preparation

The cleaning process:

- Separates CPU core and thread counts.
- Converts hardware specifications with units into numeric values.
- Converts resolution into pixel count.
- Handles missing graphics-setting labels.
- Merges CPU and GPU specifications into the FPS dataset.

Categorical features are one-hot encoded. Missing numeric features are imputed using medians. The preprocessor is fitted only on the training data and then applied to the test data.

## Model

The current model is an XGBoost regressor.

Training uses `log1p(avg_fps)` as the target. Predictions are converted back to FPS using `expm1()` before evaluation.

Current parameters:

```python
XGBRegressor(
    n_estimators=165,
    max_depth=4,
    learning_rate=0.07,
    objective='reg:squarederror',
    min_child_weight=3.3,
    subsample=0.6,
    reg_lambda=2
)
```

Linear regression, polynomial regression, and random forest are included in a separate comparison script.

## Current results

The data is split into 80% training and 20% testing using `random_state=42`.

Metric  XGBoost result

MAE   4.63
MSE   59.29
R2    0.9595
MAPE  16.73%

MAE indicates an average absolute error of approximately 4.63 FPS on the evaluated split.

The same split was inspected during model tuning, so an untouched evaluation dataset is needed for a stronger final performance estimate.

## Results and Model comparison

Linear Regression
MAE    21.10
MSE    954.45
R2     0.3474
MAPE   78.65%

Polynomial Regression
MAE   22.54
MSE   1097.56
R2    0.2496
MAPE  86.76%

Random Forest Regression
MAE   8.92
MSE   203.17
R2    0.8611
MAPE  34.70%

Linear and polynomial regression produced larger prediction errors with the settings tested. 
Random forest performed better than those baselines, while tuned XGBoost achieved the lowest errors and highest R2 on the evaluated split. 
The shape of the prediction scatter alone does not establish whether an algorithm is suitable

Only Random Forest Regression was able to compete with the model with very close and fairly accurate compare to the first two, but the MAPE is double of XGBoost and R2 is less

## Project files

 File  Purpose

 `cleaning.py`:  Loads, cleans, and merges the datasets 
 `preprocessing.py`: Selects features and creates the preprocessor 
 `train.py`: Splits the data and trains XGBoost 
 `evaluation.py`: Calculates metrics and plots predictions against actual FPS 
 `compare_models.py`: Runs linear, polynomial, and random forest experiments 
 `data/`: Contains the input CSV files 

## Running the project

Install the required packages:

```bash
python -m pip install numpy pandas matplotlib scikit-learn xgboost
```

Place these files in the `data/` folder:

- `gaming_fps_dataset.csv`
- `cpu_specifications.csv`
- `gpu_1986-2026.csv`

Train and evaluate XGBoost:

```bash
python train.py
```
The desktop interface requires Tkinter, which is included with standard Python installations on Windows. 
Run python train.py first to create models/fps_model.joblib, then launch the GUI

For GUI:

```bash
python gui.py
```
Run the other model experiments:

```bash
python compare_models.py
```

Close each plot window to allow the comparison script to continue.

## Limitations

- Matching the datasets reduced the available hardware and benchmark coverage.
- Missing hardware specifications may be imputed rather than verified.
- A random row split does not establish performance on entirely unseen CPUs, GPUs, or games.
- Predictions depend on the quality and coverage of the source data.
- Real gaming performance also depends on factors not represented here, such as RAM configuration, drivers, cooling, and game updates.

## What I learned

- Joining datasets requires consistent identifiers and careful hardware matching.
- Numeric unit conversion and missing value handling affect model inputs.
- Preprocessing must be fitted on training data.
- Log-transformed predictions must be converted back to the original units before calculating FPS errors.
- Model selection requires consistent evaluation across the same data split.
-Improve the existing desktop interface
## Future improvements

- Expand coverage with additional verified benchmark records.
- Use cross-validation for parameter tuning and reserve an untouched final test set.
- Evaluate performance on unseen hardware or game groups.
- Build an interface for selecting hardware and graphics settings.

## Data preparation and attribution

This project uses modified versions of three CPU, GPU, and gaming
FPS datasets. Original source links will be added once verified.

Preparation included standardizing hardware names, creating matching
keys, selecting relevant columns, and filtering records to hardware
available across the datasets. I wrote the initial data cleaning, model training, and evaluation code.
ChatGPT helped refine and organize the code, troubleshoot and  helped me generate the initial Tkinter interface, 
which am now learning and I integrated into the project. ChatGPT also helped me prepare
the columns and data of the datasets to make the datasets compatible.

The processed files are project specific subsets, not unchanged copies
of the original datasets. Original data ownership and licensing remain
with their respective sources.

gpu_1986-2026.csv (Likely original source, verification pending): GPUs Specs From 1986 to 2026 — ELLIMAC, Kaggle.