# monitor_drift.py
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

reference_data = pd.DataFrame(
    np.random.randn(1000, 10), 
    columns=[f"feature_{i}" for i in range(10)]
)

current_data = pd.DataFrame(
    np.random.randn(500, 10) + 0.15,
    columns=[f"feature_{i}" for i in range(10)]
)

drift_report = Report(metrics=[
    DataDriftPreset(),
])

drift_report.run(reference_data=reference_data, current_data=current_data)

drift_report.save_html("model_drift_report.html")
print("Model drift report successfully generated as 'model_drift_report.html'!")