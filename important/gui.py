from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Load your trained model and cleaned data.
model_path = (
    Path(__file__).resolve().parent
    / 'models'
    / 'fps_model.joblib'
)

if not model_path.exists():
    raise FileNotFoundError(
        'Run train.py first to create models/fps_model.joblib'
    )

saved = joblib.load(model_path)

df = saved['data'].copy()
xgb_model = saved['model']
preprocessor = saved['preprocessor']
feature_columns = saved['feature_columns']


cpu_columns = [
    'cores',
    'threads',
    'base_clock',
    'boost_clock',
    'Socket',
    'Process_Size',
    'L3_Cache',
    'TDP'
]

gpu_columns = [
    'Graphics Processor__Architecture',
    'Clock Speeds__Base Clock',
    'Clock Speeds__Boost Clock',
    'Memory__Memory Size',
    'Memory__Memory Type',
    'Memory__Memory Bus',
    'Memory__Bandwidth',
    'Board Design__TDP',
    'Top__CORES',
    'Render Config__Shading Units',
    'Render Config__RT Cores',
    'Theoretical Performance__FP32 (float)'
]

settings_columns = [
    'game',
    'resolution',
    'preset',
    'ray_tracing',
    'rt_quality',
    'upscaling',
    'upscaling_quality'
]


# Create the window.
window = tk.Tk()
window.title('Gaming FPS Predictor')

frame = ttk.Frame(window, padding=20)
frame.pack(fill='both', expand=True)

ttk.Label(
    frame,
    text='Gaming FPS Predictor',
    font=('Arial', 18, 'bold')
).grid(row=0, column=0, columnspan=2, pady=(0, 15))

dropdowns = {}

labels = {
    'cpu_name': 'CPU',
    'gpu_name': 'GPU',
    'game': 'Game',
    'resolution': 'Resolution',
    'preset': 'Graphics preset',
    'ray_tracing': 'Ray tracing',
    'rt_quality': 'Ray tracing quality',
    'upscaling': 'Upscaling',
    'upscaling_quality': 'Upscaling quality'
}

for row, (column, label) in enumerate(labels.items(), start=1):
    ttk.Label(frame, text=label).grid(
        row=row, column=0, sticky='w', padx=(0, 15), pady=5
    )

    dropdown = ttk.Combobox(
        frame,
        state='readonly',
        width=42
    )

    dropdown.grid(row=row, column=1, sticky='ew', pady=5)
    dropdowns[column] = dropdown

frame.columnconfigure(1, weight=1)

result_text = tk.StringVar(value='Choose your settings, then predict.')


def display_values(data, column):
    if column == 'ray_tracing':
        return data[column].map({0: 'Off', 1: 'On'})

    return data[column].astype(str)


# Populate the hardware dropdowns.
for column in ['cpu_name', 'gpu_name']:
    options = sorted(df[column].dropna().unique().tolist())
    dropdowns[column]['values'] = options

    if options:
        dropdowns[column].current(0)


def refresh_settings(event=None):
    available = df.copy()

    for column in settings_columns:
        values = display_values(available, column)
        options = sorted(values.dropna().unique().tolist())

        dropdown = dropdowns[column]
        previous_choice = dropdown.get()
        dropdown['values'] = options

        if previous_choice not in options:
            dropdown.set(options[0] if options else '')

        available = available[
            values == dropdown.get()
        ]

    # Clear the old prediction when settings change.
    result_text.set('Choose your settings, then predict.')


def clear_prediction(event=None):
    result_text.set('Choose your settings, then predict.')


for column in settings_columns:
    dropdowns[column].bind(
        '<<ComboboxSelected>>',
        refresh_settings
    )

for column in ['cpu_name', 'gpu_name']:
    dropdowns[column].bind(
        '<<ComboboxSelected>>',
        clear_prediction
    )


def predict_fps():
    try:
        selected_cpu = dropdowns['cpu_name'].get()
        selected_gpu = dropdowns['gpu_name'].get()

        cpu_specs = df.loc[
            df['cpu_name'] == selected_cpu,
            cpu_columns
        ].drop_duplicates()

        gpu_specs = df.loc[
            df['gpu_name'] == selected_gpu,
            gpu_columns
        ].drop_duplicates()

        if len(cpu_specs) != 1 or len(gpu_specs) != 1:
            raise ValueError(
                'Selected hardware is missing or has conflicting specs.'
            )

        # Find the selected game/settings combination.
        settings = df.copy()

        for column in settings_columns:
            values = display_values(settings, column)
            settings = settings[
                values == dropdowns[column].get()
            ]

        if settings.empty:
            raise ValueError('No matching game settings were found.')

        metadata = settings[
            ['engine', 'demand_score']
        ].drop_duplicates()

        if len(metadata) != 1:
            raise ValueError(
                'This game/settings combination has conflicting metadata.'
            )

        # Select only model inputs, excluding the recorded avg_fps.
        input_row = (
            settings.loc[:, feature_columns]
            .iloc[[0]]
            .copy()
        )

        # Substitute the chosen CPU and GPU specifications.
        for column in cpu_columns:
            input_row[column] = cpu_specs[column].iloc[0]

        for column in gpu_columns:
            input_row[column] = gpu_specs[column].iloc[0]

        input_encoded = preprocessor.transform(input_row)

        predicted_fps = np.expm1(
            xgb_model.predict(input_encoded)
        )[0]

        if not np.isfinite(predicted_fps) or predicted_fps < 0:
            raise ValueError('The model returned an invalid FPS estimate.')

        result_text.set(
            f'Estimated average FPS: {predicted_fps:.1f}'
        )

    except Exception as error:
        result_text.set('Prediction could not be completed.')
        messagebox.showerror('Prediction error', str(error))


# Optional evaluation graph.
show_graph = tk.BooleanVar(value=False)
graph_window = None


def close_graph():
    global graph_window

    if graph_window is not None:
        graph_window.destroy()
        graph_window = None

    show_graph.set(False)


def toggle_graph():
    global graph_window

    if not show_graph.get():
        close_graph()
        return

    graph_window = tk.Toplevel(window)
    graph_window.title('Model evaluation')
    graph_window.protocol('WM_DELETE_WINDOW', close_graph)

    actual = np.asarray(saved['y_test'])
    predicted = np.asarray(saved['predictions'])

    figure = Figure(figsize=(6, 4), tight_layout=True)
    ax = figure.add_subplot(111)

    ax.scatter(predicted, actual, color='green')

    minimum = min(predicted.min(), actual.min())
    maximum = max(predicted.max(), actual.max())

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        color='red',
        linestyle='--',
        label='Perfect prediction'
    )

    ax.set_xlabel('Predicted FPS')
    ax.set_ylabel('Actual FPS')
    ax.set_title('XGBoost: saved test-set predictions')
    ax.legend()

    canvas = FigureCanvasTkAgg(figure, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # Keep the canvas available while the graph window is open.
    graph_window.canvas = canvas


ttk.Button(
    frame,
    text='Predict FPS',
    command=predict_fps
).grid(row=10, column=0, columnspan=2, pady=15)

ttk.Label(
    frame,
    textvariable=result_text,
    font=('Arial', 12, 'bold')
).grid(row=11, column=0, columnspan=2, pady=5)

ttk.Checkbutton(
    frame,
    text='Show model evaluation graph',
    variable=show_graph,
    command=toggle_graph
).grid(row=12, column=0, columnspan=2, pady=10)

ttk.Label(
    frame,
    text=(
        'Settings are limited to combinations recorded for each game.\n'
        'FPS is an estimate; new hardware combinations may be less reliable.'
    ),
    wraplength=480
).grid(row=13, column=0, columnspan=2, pady=5)

refresh_settings()

window.mainloop()