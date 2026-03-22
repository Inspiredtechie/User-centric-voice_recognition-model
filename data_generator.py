import pyaudio
import wave
import os
import time

# CONFIGURATION
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 2  # Length of each sample
OWNER_NAME = "Owner"
DATASET_PATH = "dataset"

def record_samples(sample_type, num_samples):
    audio = pyaudio.PyAudio()
    path = os.path.join(DATASET_PATH, sample_type)
    if not os.path.exists(path):
        os.makedirs(path)

    print(f"\n[INFO] Prepare to record {num_samples} samples for '{sample_type}'.")
    print("[INFO] Press 'Enter' to start recording continuously...")
    input()

    for i in range(num_samples):
        print(f"Recording {i+1}/{num_samples}...", end="\r")
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK)
        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        # Save the file
        filename = f"{sample_type}_{int(time.time())}_{i}.wav"
        wf = wave.open(os.path.join(path, filename), 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        time.sleep(0.5) # Short pause between samples

    audio.terminate()
    print(f"\n[SUCCESS] recording for {sample_type} complete.")

if __name__ == "__main__":
    # 1. Record Owner Voice
    record_samples(OWNER_NAME, 10) 
    
    # 2. Record/Gather 'Unknown' Voice
    # Crucial: You must record other people, TV noise, or silence as "Unknown" 
    # for the model to learn the difference.
    print("\n[IMPORTANT] Now recording 'Unknown' voices (Background/Others).")
    print("Play TV, music, or have friends speak.")
    record_samples("Unknown", 10)