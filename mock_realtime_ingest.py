import asyncio
import websockets
import json
import time
import random
import pandas as pd
import threading
import sys
from datetime import datetime

# Global flag for black swan injection
inject_black_swan_flag = False

def listen_for_black_swan():
    global inject_black_swan_flag
    while True:
        try:
            # Wait for user to type 'b' and hit enter
            user_input = input()
            if user_input.strip().lower() == 'b':
                inject_black_swan_flag = True
                print("\n[!] BLACK SWAN TRIGGERED: Will inject on next cycle.\n")
        except EOFError:
            break

async def simulate_realtime_stream():
    global inject_black_swan_flag
    print("=" * 60)
    print("GRIDLOCK CHRONOLOGICAL TELEMETRY REPLAY")
    print("=" * 60)
    
    WS_URL = "ws://localhost:8000/ws/live"
    speed_multiplier = 60.0  # 1 real minute = 1 second of simulation
    
    try:
        df = pd.read_parquet("data/clean_unplanned.parquet")
        # Ensure we have start_datetime and sort chronologically
        df['start_datetime'] = pd.to_datetime(df['start_datetime'], errors='coerce')
        df = df.dropna(subset=['start_datetime']).sort_values('start_datetime')
        print(f"Loaded {len(df)} historical events for chronological replay.")
    except Exception as e:
        print(f"Could not load parquet data ({e}). Please ensure data pipeline has run.")
        return

    # Start keyboard listener in background
    print("\n[INFO] Type 'b' and press Enter at any time to inject a Black Swan event.\n")
    bg_thread = threading.Thread(target=listen_for_black_swan, daemon=True)
    bg_thread.start()
        
    print(f"Connecting to {WS_URL}...")
    print("Press Ctrl+C to stop.\n")
    
    while True:
        try:
            async with websockets.connect(WS_URL) as websocket:
                print(f"Connected to {WS_URL}!")
                
                async def listen():
                    try:
                        while True:
                            message = await websocket.recv()
                            payload = json.loads(message)
                            if payload.get("type") == "NEW_PREDICTION":
                                res = payload["data"]
                                if isinstance(res, list): res = res[0]
                                print(f"    <- Forecast Duration: {res.get('pred_duration_bucket', 'Unknown')}")
                                print(f"    <- Action: {res.get('manpower', 'None')}")
                                print(f"    <- Routing Plan: {res.get('diversion', 'None')}")
                                print(f"    <- Confidence: {res.get('confidence_score', 'N/A')}")
                    except websockets.exceptions.ConnectionClosed:
                        pass
                
                listen_task = asyncio.create_task(listen())
                
                prev_time = None
                
                for _, row in df.iterrows():
                    curr_time = row['start_datetime']
                    
                    # Handle Black Swan Injection
                    if inject_black_swan_flag:
                        incident = {
                            "id": f"BS_{int(time.time())}",
                            "timestamp": curr_time.isoformat() if pd.notna(curr_time) else datetime.now().isoformat(),
                            "latitude": 12.9716, # City center
                            "longitude": 77.5946,
                            "description": "MAJOR CHEMICAL SPILL AND FIRE MULTIPLE CASUALTIES ROAD BLOCKED",
                            "hour": int(time.localtime().tm_hour),
                            "day_of_week": int(time.localtime().tm_wday),
                            "corridor": "UNSEEN_NEW_CORRIDOR", # Forces GNN fallback
                            "priority": "High",
                            "requires_road_closure": True,
                            "event_cause": "chemical spill"
                        }
                        print(f"\n[!!!] INJECTING BLACK SWAN -> {incident['id']} on {incident['corridor']} ({incident['priority']} Priority)")
                        await websocket.send(json.dumps({"incidents": [incident]}))
                        inject_black_swan_flag = False
                        await asyncio.sleep(2)
                    
                    # Normal chronological event
                    incident = {
                        "id": str(row.get('kgid', f"live_{random.randint(1000, 9999)}")), # Try to use real ID if exists
                        "timestamp": curr_time.isoformat() if pd.notna(curr_time) else datetime.now().isoformat(),
                        "latitude": float(row['latitude']) if pd.notna(row.get('latitude')) else 12.9716,
                        "longitude": float(row['longitude']) if pd.notna(row.get('longitude')) else 77.5946,
                        "description": str(row.get('description', 'Traffic buildup detected by CCTV')),
                        "hour": int(row.get('hour', time.localtime().tm_hour)),
                        "day_of_week": int(row.get('day_of_week', time.localtime().tm_wday)),
                        "corridor": str(row.get('corridor', 'Unknown')),
                        "priority": str(row.get('priority', 'Medium')),
                        "requires_road_closure": bool(row.get('requires_road_closure', False))
                    }
                    
                    if prev_time is not None:
                        diff_seconds = (curr_time - prev_time).total_seconds()
                        if diff_seconds > 0:
                            # Compress time
                            sleep_time = min(diff_seconds / speed_multiplier, 10.0) # Cap at 10s wait
                            print(f"(Replaying time... waiting {sleep_time:.1f}s)")
                            await asyncio.sleep(sleep_time)
                    
                    prev_time = curr_time
                        
                    print(f"[+] Streaming Telemetry -> {incident['id']} on {incident['corridor']} ({incident['priority']} Priority) at {curr_time}")
                    
                    await websocket.send(json.dumps({"incidents": [incident]}))
                    await asyncio.sleep(0.5) # Give API a moment to process before next tight loop iteration
                    
                print("Reached end of dataset replay. Restarting...")
                    
        except websockets.exceptions.ConnectionClosedError:
            print("    [!] Connection closed. Reconnecting in 5s...")
            await asyncio.sleep(5)
        except ConnectionRefusedError:
            print("    [!] Failed to connect to API. Is the server running on port 8000?")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"    [!] Error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(simulate_realtime_stream())
    except KeyboardInterrupt:
        print("\nStopped mock telemetry stream.")
