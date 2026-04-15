import os
import json
import random
from datetime import datetime, timedelta

APPS = [
    "paytm", "mobikwik", "freecharge", "google_pay", "mcd_app", 
    "mseva", "bses_rajdhani", "mahadiscom", "mpassport_seva", 
    "aarogya_setu", "digilocker", "coinswitch", "wazirx", 
    "coindcx", "binance"
]

IDENTITY = {
    "name": "Aarav Sharma",
    "phone": "+91-9876543210",
    "email": "aarav.sharma@example.com",
    "aadhaar": "1234-5678-9012",
    "pan": "ABCDE1234F",
    "bank_account": "HDFC0001234",
    "login_id": "aarav_s_88",
    "password": "hashed_pw_99x"
}

TELEMETRY = {
    "local_ip": "192.168.1.45",
    "device_id": "SM-G991B-SEC",
    "os_version": "Android 14.0.1",
    "ram_status": "12GB Total / 4.2GB Free",
    "storage_status": "256GB / 120GB Free",
    "network_status": "Airtel_Home_WiFi",
    "battery_percentage": "87%",
    "advertisement_id": "550e8400-e29b-41d4-a716-446655440000"
}

def generate_balanced_data():
    base_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    os.makedirs("fake_data", exist_ok=True)
    all_findings = []
    global_id = 1

    for app in APPS:
        app_name_display = app.replace('_', ' ').title()
        if app == "google_pay": app_name_display = "Google Pay"
        if app == "mcd_app": app_name_display = "MCD App"
        
        filename = f"fake_data/{app}.jsonl"
        with open(filename, "w") as f:
            for i in range(1, 51):
                is_compliant = (i % 2 == 0)
                finding_id = f"F-{app[:3].upper()}-{'C' if is_compliant else 'V'}-{str(i).zfill(3)}"
                timestamp = (base_time - timedelta(days=random.randint(0, 30), hours=random.randint(1, 23))).isoformat() + "+05:30"
                
                record = {
                    "finding_id": finding_id,
                    "app_name": app_name_display,
                    "action": "data collection",
                    "destination": "Internal Server" if is_compliant else "Third-party Node",
                    "timestamp": timestamp
                }
                
                if is_compliant:
                    record["collection_context"] = random.choice(["User Account Registration", "Profile Update", "Payment Transaction", "Support Request"])
                    
                    # Use only 1 or 2 extremely common fields so the Strict LLM is guaranteed to find explicit authorization
                    c_type = random.choice(["phone", "email", "name"])
                    record[f"collected_{c_type}"] = IDENTITY[c_type]
                    
                    if random.random() > 0.5:
                        other_type = random.choice([k for k in ["phone", "email", "name"] if k != c_type])
                        record[f"collected_{other_type}"] = IDENTITY[other_type]
                else:
                    record["collection_context"] = random.choice(["Background Technical Telemetry", "App Idle State", "Changing App Theme"])
                    record["collected_device_id"] = TELEMETRY["device_id"]
                    record["collected_os_version"] = TELEMETRY["os_version"]
                    
                    violations = ["collected_aadhaar", "collected_pan", "collected_bank_account", 
                                  "collected_ram_status", "collected_storage_status", "collected_local_ip", 
                                  "collected_battery_percentage", "collected_password", "collected_network_status", "collected_advertisement_id"]
                    for _ in range(random.randint(2, 4)):
                        v = random.choice(violations)
                        if "aadhaar" in v: record[v] = IDENTITY["aadhaar"]
                        elif "pan" in v: record[v] = IDENTITY["pan"]
                        elif "bank" in v: record[v] = IDENTITY["bank_account"]
                        elif "password" in v: record[v] = IDENTITY["password"]
                        else: record[v] = TELEMETRY[v.replace("collected_", "")]

                f.write(json.dumps(record) + "\n")
                all_findings.append(record)
                global_id += 1

    with open("fake_data.jsonl", "w") as mf:
        for record in all_findings:
            mf.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    generate_balanced_data()
    print("Strictly research-aligned balanced fake data generated.")
