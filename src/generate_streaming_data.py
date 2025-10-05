import os
import random
import time
from datetime import datetime
import pandas as pd
import numpy as np

output_dir = "C:/Project/Fitness Tracker Analysis/data_lake/streaming_input"
activities = ["walking", "running", "cycling", "swimming", "yoga", "gym_workout", "hiking"]

def generate_batch():
    """Generate a single batch of realistic fitness data"""
    records = []
    # Generate 5-10 new records
    for _ in range(random.randint(5, 10)):
        activity = random.choice(activities)
        user_id = random.randint(1, 2000)  # Simulate for our existing user base
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # More realistic data generation based on activity type
        if activity == "walking":
            steps = np.random.randint(50, 200)
            calories_burned = steps * 0.04
            heart_rate_avg = np.random.randint(70, 110)
        elif activity == "running":
            steps = np.random.randint(200, 500)
            calories_burned = steps * 0.08
            heart_rate_avg = np.random.randint(120, 160)
        elif activity == "cycling":
            steps = np.random.randint(100, 300)
            calories_burned = steps * 0.06
            heart_rate_avg = np.random.randint(100, 140)
        elif activity == "swimming":
            steps = np.random.randint(10, 50)
            calories_burned = np.random.randint(200, 600)
            heart_rate_avg = np.random.randint(110, 150)
        elif activity == "yoga":
            steps = np.random.randint(5, 50)
            calories_burned = np.random.randint(100, 300)
            heart_rate_avg = np.random.randint(60, 100)
        elif activity == "gym_workout":
            steps = np.random.randint(50, 150)
            calories_burned = np.random.randint(300, 800)
            heart_rate_avg = np.random.randint(120, 160)
        else:  # hiking
            steps = np.random.randint(100, 400)
            calories_burned = steps * 0.07
            heart_rate_avg = np.random.randint(110, 150)
        
        sleep_hours = np.random.uniform(6, 9)

        records.append([
            user_id, timestamp, steps,
            round(calories_burned, 2), heart_rate_avg, round(sleep_hours, 2), activity
        ])

    df = pd.DataFrame(records, columns=[
        "user_id", "timestamp", "steps", "calories_burned", "heart_rate_avg", "sleep_hours", "activity_type"
    ])

    return df

def main():
    print("Starting live data stream simulation...")
    print(f"Output directory: {output_dir}")
    print("Press Ctrl+C to stop")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    file_counter = 0
    try:
        while True:
            df = generate_batch()
            
            # Save as a single Parquet file with a unique name
            file_path = os.path.join(output_dir, f"data_{file_counter}_{int(time.time())}.parquet")
            df.to_parquet(file_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Generated file: data_{file_counter}_{int(time.time())}.parquet with {len(df)} records")

            file_counter += 1
            time.sleep(10)  # Generate new data every 10 seconds
            
    except KeyboardInterrupt:
        print("\n✓ Data generation stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
