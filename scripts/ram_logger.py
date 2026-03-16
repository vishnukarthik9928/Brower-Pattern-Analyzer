# src/collect/system_resource_monitor.py
"""
Continuously records system memory usage, browser memory usage,
and CPU load while the user is browsing.

Run this script in a separate terminal before starting browsing.

Example:
    python src/collect/system_resource_monitor.py
"""

import psutil
import pandas as pd
import yaml
import time
import os
from datetime import datetime


def monitor_resources(run_minutes=60, check_interval=10, save_file="data/system_usage.csv"):
    """
    Track RAM usage, browser memory consumption, and CPU load.

    Parameters
    ----------
    run_minutes : int
        Total runtime for monitoring.
    check_interval : int
        Time gap between each measurement.
    save_file : str
        CSV file path for saving collected metrics.
    """

    os.makedirs("data", exist_ok=True)

    collected_data = []
    finish_time = time.time() + (run_minutes * 60)
    expected_records = int((run_minutes * 60) / check_interval)
    iteration = 0

    print(f"📈 System resource monitor started")
    print(f"   Duration: {run_minutes} minutes")
    print(f"   Interval: {check_interval} seconds")
    print(f"   Expected records: ~{expected_records}")
    print(f"   Output file: {save_file}")
    print("   Press Ctrl+C to terminate early.\n")

    try:
        while time.time() < finish_time:

            memory_info = psutil.virtual_memory()
            total_browser_memory = 0.0
            detected_browser_processes = []

            # Scan running processes
            for process in psutil.process_iter(["name", "memory_info", "pid"]):
                try:
                    process_name = process.info["name"]

                    if process_name in [
                        "chrome.exe",
                        "msedge.exe",
                        "Google Chrome",
                        "Microsoft Edge"
                    ]:
                        memory_mb = process.info["memory_info"].rss / (1024 ** 2)
                        total_browser_memory += memory_mb
                        detected_browser_processes.append(process.info["pid"])

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Create log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "system_ram_used_mb": round(memory_info.used / (1024 ** 2), 1),
                "system_ram_free_mb": round(memory_info.available / (1024 ** 2), 1),
                "system_ram_total_mb": round(memory_info.total / (1024 ** 2), 1),
                "system_ram_percent": memory_info.percent,
                "browser_memory_mb": round(total_browser_memory, 1),
                "browser_process_count": len(detected_browser_processes),
                "cpu_usage_percent": psutil.cpu_percent(interval=1)
            }

            collected_data.append(log_entry)
            iteration += 1

            print(
                f"[{iteration}/{expected_records}] "
                f"RAM: {log_entry['system_ram_percent']}% | "
                f"Browser RAM: {log_entry['browser_memory_mb']:.0f} MB | "
                f"CPU: {log_entry['cpu_usage_percent']}%",
                end="\r"
            )

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print(f"\n\n⚠ Monitoring stopped early after {iteration} records.")

    # Save collected data
    df = pd.DataFrame(collected_data)
    df.to_csv(save_file, index=False)

    print(f"\n✔ Data saved ({len(df)} rows) → {save_file}")

    return df


if __name__ == "__main__":

    # Load configuration settings
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    monitor_resources(
        run_minutes=config["ram_logger"]["duration_minutes"],
        check_interval=config["ram_logger"]["interval_seconds"],
        save_file=config["paths"]["ram_log"]
    )
