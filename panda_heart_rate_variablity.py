import pandas as pd
import numpy as np
import matplotlib
# Headless-safe plotting for RPi / SSH
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ======================================================
# 1. Load or generate heart rate data
# ======================================================
def load_hr_data(source="synthetic", csv_path=None, n_minutes=1440, seed=42):
    """
    source:
        - 'synthetic': generate test HR data
        - 'csv'      : load CSV with columns ['timestamp','HR']
    """
    if source == "csv" and csv_path:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        return df[["HR"]]
    
    # fallback: synthetic
    np.random.seed(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_minutes, freq="T")
    base_hr = 70
    circadian = 10 * np.sin(2 * np.pi * timestamps.hour / 24)  # daily rhythm
    noise = np.random.normal(0, 5, n_minutes)
    hr = base_hr + circadian + noise
    df = pd.DataFrame({"HR": hr}, index=timestamps)
    return df

# ======================================================
# 2. Compute HRV metrics
# ======================================================
def compute_hrv(df, window=5):
    df = df.copy()
    df["HRV"] = df["HR"].diff().abs()
    df["HRV_avg"] = df["HRV"].rolling(window).mean()
    return df

# ======================================================
# 3. Statistical analysis
# ======================================================
def analyze_hrv(df):
    analysis = {
        "HR_mean": df["HR"].mean(),
        "HR_std": df["HR"].std(),
        "HR_min": df["HR"].min(),
        "HR_max": df["HR"].max(),
        "HRV_mean": df["HRV"].mean(),
        "HRV_std": df["HRV"].std(),
        "HRV_min": df["HRV"].min(),
        "HRV_max": df["HRV"].max()
    }
    return analysis

# ======================================================
# 4. Save figures
# ======================================================
def save_figures(df, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Heart Rate plot
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["HR"], label="HR")
    plt.title("Heart Rate (BPM)")
    plt.xlabel("Time")
    plt.ylabel("BPM")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"HR_{timestamp}.png", dpi=150)
    plt.close()

    # HRV plot
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["HRV"], label="HRV")
    plt.plot(df.index, df["HRV_avg"], label="HRV_avg", color="red")
    plt.title("Heart Rate Variability (BPM difference)")
    plt.xlabel("Time")
    plt.ylabel("BPM diff")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"HRV_{timestamp}.png", dpi=150)
    plt.close()

    print(f"[OK] Figures saved to '{out_dir}'")

# ======================================================
# 5. Save textual report
# ======================================================
def save_report(analysis, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"HRV_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write("=== HR & HRV STATISTICS ===\n")
        for k, v in analysis.items():
            f.write(f"{k}: {v:.2f}\n")
    print(f"[OK] Report saved to '{report_path}'")

# ======================================================
# 6. Optional alerts for abnormal HR or HRV
# ======================================================
def check_alerts(df, hr_max=100, hr_min=50, hrv_max=20):
    alerts = []
    if df["HR"].max() > hr_max:
        alerts.append(f"High HR detected: {df['HR'].max():.1f} bpm")
    if df["HR"].min() < hr_min:
        alerts.append(f"Low HR detected: {df['HR'].min():.1f} bpm")
    if df["HRV"].max() > hrv_max:
        alerts.append(f"High HRV detected: {df['HRV'].max():.1f} bpm diff")
    return alerts

# ======================================================
# 7. Main pipeline
# ======================================================
def main():
    # Load synthetic or real HR data
    df = load_hr_data(source="synthetic")  # change to 'csv' and provide path for real data

    # Compute HRV
    df = compute_hrv(df, window=5)

    # Analyze statistics
    analysis = analyze_hrv(df)
    print("HR & HRV Analysis:", analysis)

    # Optional alerts
    alerts = check_alerts(df)
    for alert in alerts:
        print("[ALERT]", alert)

    # Save figures and report (RPi-friendly)
    save_figures(df)
    save_report(analysis)

if __name__ == "__main__":
    main()
