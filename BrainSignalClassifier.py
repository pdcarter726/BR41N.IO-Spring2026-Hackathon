import time
import numpy as np
import mss
import json
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
# We use a wider box to capture the whole pattern at once.
# This captures 150 pixels of the wave right before it hits the control panel.
MONITOR = {"top": 180, "left": 1350, "width": 250, "height": 60} 
BASELINE_Y = 30 
COOLDOWN_FRAMES = 30 

def extract_features(wave_array):
    """Transforms the physical shape of the line into ML features."""
    data = np.array(wave_array)
    
    # How tall/deep is the pattern?
    amplitude = np.max(data) - np.min(data)
    
    # How much does it deviate from the flatline?
    variance = np.var(data)
    
    # How "jagged" is it? (Sum of point-to-point differences)
    differences = np.abs(np.diff(data))
    jaggedness = np.sum(differences)
    
    return [amplitude, variance, jaggedness]

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

def train_real_model():
    """Loads recorded data and trains the Random Forest."""
    print("Loading training data...")
    try:
        with open("training_data.json", "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("ERROR: Could not find 'training_data.json'. Run the collector script first!")
        exit()
        
    X_train = []
    y_train = []
    
    # Loop through the JSON file and extract the features for every recorded wave
    for label, waves in dataset.items():
        for wave in waves:
            # Convert the raw line shape into ML features (amplitude, variance, jaggedness)
            features = extract_features(wave)
            X_train.append(features)
            y_train.append(label)

    print("Training ML Model on your actual data...")
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    print("Model Trained! Ready to control the robot.\n")
    
    return clf

def main():
    clf = train_real_model()
    sct = mss.mss()
    cooldown = 0
    
    try:
        while True:
            # 1. Grab the wide screen slice
            img = np.array(sct.grab(MONITOR))
            
            # 2. Trace the line to get the physical pattern
            current_wave = get_wave_snapshot(img)
            
            # 3. ML Classification Logic
            if cooldown > 0:
                cooldown -= 1
            else:
                features = extract_features(current_wave)

                # DEBUG PRINT: Show the amplitude in the console for real-time feedback
                print(f"Live Amplitude: {features[0]}")
                
                # Check if the amplitude is high enough to be an intentional movement
                if features[0] > 40: 
                    prediction = clf.predict([features])[0]
                    
                    if prediction == "ROTATE":
                        print("🦷 JAW CLENCH DETECTED! -> Robot will ROTATE")
                        cooldown = COOLDOWN_FRAMES
                        
                    elif prediction == "MOVE":
                        print("🤨 SHOULDER SHRUG DETECTED! -> Robot will MOVE")
                        cooldown = COOLDOWN_FRAMES
                        
                    # (If prediction is "NEUTRAL", it just does nothing and stays quiet)
            
            time.sleep(1/60)

    except KeyboardInterrupt:
        print("\nExiting program.")

if __name__ == "__main__":
    main()