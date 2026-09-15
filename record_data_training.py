import time
import json
import numpy as np
import mss

# --- MUST MATCH YOUR MAIN SCRIPT ---
MONITOR = {"top": 180, "left": 1350, "width": 250, "height": 60} 
BASELINE_Y = 30

def get_wave_snapshot(img_array):
    """Traces the line by finding the brightest pixel in each vertical column."""
    wave = []
    for x in range(img_array.shape[1]):
        # Extract the Blue channel for this specific column
        column = img_array[:, x, 0] 
        
        # Find the brightness value of the brightest pixel in the column
        max_brightness = np.max(column)
        
        # As long as the pixel is brighter than dark grey (50), we found the line!
        if max_brightness > 50:
            # np.argmax gives us the exact Y-coordinate of that brightest pixel
            y_val = int(np.argmax(column))
            wave.append(y_val)
        else:
            # Fallback if the line disappears completely
            wave.append(wave[-1] if wave else BASELINE_Y)
            
    return wave

def record_action(action_name, num_samples, sct):
    """Handles the countdown and recording loop."""
    print(f"\n=== RECORDING: {action_name} ===")
    print(f"We will record {num_samples} samples. Press Enter when ready to begin.")
    input()
    
    samples = []
    for i in range(num_samples):
        print(f"[{i+1}/{num_samples}] Get ready...")
        time.sleep(2) # 2-second warning
        
        print(f"--> {action_name.upper()} NOW! <--")
        # Wait 0.5 seconds for the wave to scroll left into our 150px box
        time.sleep(0.5) 
        
        # Take the snapshot
        img = np.array(sct.grab(MONITOR))
        wave = get_wave_snapshot(img)
        samples.append(wave)
        print("Captured!\n")
        
    return samples

def main():
    sct = mss.mss()
    dataset = {}
    
    # We record 20 samples of each state. 
    # (20 is usually plenty for a Random Forest to learn obvious patterns)
    dataset["NEUTRAL"] = record_action("Neutral (Just sit still)", 100, sct)
    dataset["MOVE"] = record_action("Shoulder Shrug", 100, sct)
    dataset["ROTATE"] = record_action("Jaw Clench", 100, sct)
    
    # Save all the recorded arrays to a file
    with open("training_data.json", "w") as f:
        json.dump(dataset, f)
        
    print("\nSUCCESS! Dataset saved to 'training_data.json'.")

if __name__ == "__main__":
    main()