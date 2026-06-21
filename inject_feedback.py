import requests
import sys

def main():
    print("=== Gridlock Feedback Injection ===")
    incident_id = input("Enter Incident ID (e.g. live_1234 or real kgid): ")
    duration = input("Enter Actual Duration in Minutes: ")
    corridor = input("Enter Corridor Name (optional): ")
    cause = input("Enter Event Cause (optional): ")
    
    try:
        duration_float = float(duration)
    except ValueError:
        print("Duration must be a number.")
        sys.exit(1)
        
    payload = {
        "id": incident_id,
        "actual_duration_min": duration_float,
        "corridor": corridor if corridor else "Unknown",
        "event_cause": cause if cause else "Unknown"
    }
    
    try:
        print(f"\nSending feedback for {incident_id}...")
        resp = requests.post("http://localhost:8000/api/feedback", json=payload)
        resp.raise_for_status()
        print("Success:", resp.json())
    except Exception as e:
        print("Error sending feedback:", e)
        
if __name__ == "__main__":
    main()
