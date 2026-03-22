import pyaudio
import time
from classifier import VoiceClassifier

# SETTINGS
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
BUFFER_SECONDS = 2.5 # Listen in 2.5 second windows

def main():
    classifier = VoiceClassifier()
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("\n[SYSTEM] Listening... (Speak now)")
    print("-" * 50)

    try:
        while True:
            frames = []
            # Read chunks for BUFFER_SECONDS
            for _ in range(0, int(RATE / CHUNK * BUFFER_SECONDS)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            audio_buffer = b''.join(frames)
            
            # Predict
            speaker, scores = classifier.predict(audio_buffer, RATE)
            
            # Formatting Output
            if speaker == "Soumar":
                print(f"\033[92m >> Voice recognised as {speaker} \033[0m  (Score: {scores['Soumar']:.2f})")
                # INSERT YOUR ASSISTANT TRIGGER CODE HERE
                # e.g., if "keyword" in speech_to_text(audio_buffer): execute_command()
            else:
                print(f"\033[91m >> Unknown voice \033[0m (Identified as {speaker})")

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopping...")
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()