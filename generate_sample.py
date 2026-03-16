# src/collect/generate_sample.py
"""
Creates synthetic browsing history and RAM usage data for testing
the browser analyzer project without using real browser data.
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# -----------------------------
# Domain → Category Mapping
# -----------------------------
category_map = {

    # Learning
    "chatgpt.com": "learning",
    "claude.ai": "learning",
    "claude.com": "learning",
    "365datascience.com": "learning",
    "w3schools.com": "learning",
    "guvi.in": "learning",
    "sqlite.org": "learning",
    "dev.mysql.com": "learning",
    "mysql.com": "learning",
    "zenclass.in": "learning",
    "v2.zenclass.in": "learning",
    "classify.zenclass.in": "learning",

    # Coding / Development
    "github.com": "coding",
    "raw.githubusercontent.com": "coding",
    "colab.research.google.com": "coding",
    "localhost:8501": "coding",
    "sqlitebrowser.org": "coding",

    # Work / Productivity
    "linkedin.com": "work",
    "naukri.com": "work",
    "indeed.com": "work",
    "docs.google.com": "work",
    "drive.google.com": "work",
    "forms.gle": "work",

    # Social / Communication
    "x.com": "social",
    "reddit.com": "social",
    "web.whatsapp.com": "social",
    "meet.google.com": "social",
    "mail.google.com": "social",

    # Entertainment
    "youtube.com": "entertainment",
    "netflix.com": "entertainment",

    # Utility / General
    "google.com": "utility",
    "amazon.in": "utility",
    "png2jpg.com": "utility",
    "chromewebstore.google.com": "utility",
    "tidbcloud.com": "utility",
    "auth.tidbcloud.com": "utility",
    "pingcap.com": "utility",
    "nvidia.com": "utility",
    "support.microsoft.com": "utility",
    "go.microsoft.com": "utility",
    "accounts.google.com": "utility",
    "accounts.google.co.in": "utility",
    "accounts.youtube.com": "utility",
    "test.hyrenet.in": "utility"
}

domains = list(category_map.keys())

# Hourly browsing probability weights
hour_weights = [
    0.2,0.1,0.05,0.05,0.05,0.1,
    0.3,0.8,1.2,1.5,1.8,1.5,
    1.8,1.2,1.0,1.2,1.5,2.0,
    2.5,2.8,2.5,1.8,1.2,0.8
]


# -----------------------------
# Generate Browsing History
# -----------------------------
def generate_browsing_history(days=5, records_per_day=120):

    print(f"Generating browsing history for {days} days...")

    rows = []
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for d in range(days):

        date = base_date - timedelta(days=(days-d))
        n = records_per_day + random.randint(-20,20)

        hours = random.choices(range(24), weights=hour_weights, k=n)
        hours.sort()

        for hour in hours:

            minute = random.randint(0,59)
            second = random.randint(0,59)

            timestamp = date + timedelta(hours=hour,minutes=minute,seconds=second)

            domain = random.choice(domains)

            rows.append({
                "timestamp": timestamp.isoformat(),
                "domain": domain,
                "category": category_map[domain],
                "title": f"Page on {domain}",
                "hour": hour,
                "date": date.date(),
                "day_name": date.strftime("%A")
            })

    df = pd.DataFrame(rows).sort_values("timestamp")

    os.makedirs("data", exist_ok=True)

    df.to_csv("data/browsing_history.csv",index=False)

    print(f"Saved {len(df)} browsing records")

    return df


# -----------------------------
# Generate RAM Usage Log
# -----------------------------
def generate_ram_log(days=5, interval_seconds=10):

    print("Generating RAM usage log...")

    rows = []

    base = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
    start = base - timedelta(days=days)

    current = start
    end = base

    base_system_ram = 6000
    base_browser_ram = 800

    while current <= end:

        hour = current.hour

        if 18 <= hour <= 23:
            spike = random.uniform(1.0,1.6)
        elif 9 <= hour <= 17:
            spike = random.uniform(1.0,1.4)
        else:
            spike = random.uniform(0.7,1.1)

        system_ram = base_system_ram * spike + random.gauss(0,100)
        browser_ram = base_browser_ram * spike + random.gauss(0,80)

        browser_ram = max(200,browser_ram)
        system_ram = max(3000,min(system_ram,15000))

        rows.append({
            "timestamp": current.isoformat(),
            "ram_used_mb": round(system_ram,1),
            "ram_available_mb": round(16384-system_ram,1),
            "browser_ram_mb": round(browser_ram,1),
            "cpu_percent": round(random.uniform(5,80)*spike,1)
        })

        current += timedelta(seconds=interval_seconds)

    df = pd.DataFrame(rows)

    df.to_csv("data/ram_log.csv",index=False)

    print(f"Saved {len(df)} RAM records")

    return df


# -----------------------------
# Export Domain Category Map
# -----------------------------
def generate_domain_category_map():

    rows = [{"domain":d,"category":c} for d,c in category_map.items()]

    df = pd.DataFrame(rows)

    df.to_csv("data/domain_category_map.csv",index=False)

    print("Saved domain category mapping")

    return df


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    print("="*50)
    print("Generating Sample Browser Dataset")
    print("="*50)

    generate_browsing_history(days=5)
    generate_ram_log(days=5)
    generate_domain_category_map()

    print("\nAll sample datasets generated successfully!")