import os
import json
from datetime import datetime, timedelta

def enhance_data():
    base_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    
    with open("fake_data.jsonl", "r") as f:
        lines = f.readlines()
        
    os.makedirs("fake_data", exist_ok=True)
    
    # Clear out the per-app files
    app_files = {}

    for line in lines:
        if not line.strip(): continue
        
        record = json.loads(line.strip())
        
        # 1. Update timestamp to Indian Timings
        # Randomize the day slightly but keep the +05:30
        record["timestamp"] = (base_time - timedelta(days=1)).isoformat() + "+05:30"
        
        # 2. Ensure data_type is intrinsically part of the collected footprint bundle
        # So "data_type": "Password" becomes "collected_password": "XXXX"
        dtype = record.get("data_type", "")
        if dtype:
            slug = dtype.lower().replace(" ", "_")
            if f"collected_{slug}" not in record:
                if "password" in slug:
                    record[f"collected_{slug}"] = "hashed_pw_99x"
                elif "battery" in slug:
                    record[f"collected_{slug}"] = "87%"
                elif "device" in slug or "imei" in slug:
                    record[f"collected_{slug}"] = "SM-G991B-SEC"
                elif "network" in slug or "wifi" in slug:
                    record[f"collected_{slug}"] = "Airtel_Home_WiFi"
                elif "location" in slug or "gps" in slug:
                    record[f"collected_{slug}"] = "Lat 28.70, Lon 77.10"
                else:
                    record[f"collected_{slug}"] = "[Sensitive Data]"
        
        # 3. Save back into per-app mapping
        app_name = record["app_name"].lower().replace(" ", "_")
        if app_name not in app_files:
            app_files[app_name] = []
        app_files[app_name].append(record)

    # Write per-app
    for app_name, records in app_files.items():
        with open(f"fake_data/{app_name}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
                
    # Overwrite the mega file with enhanced data
    with open("fake_data.jsonl", "w") as mf:
        for app_name, records in app_files.items():
            for r in records:
                mf.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    enhance_data()
    print("Faithfully enhanced research data mapping completed.")
