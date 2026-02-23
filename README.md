# realtime-emotion-detection
pip install deepface transformers torch librosa sounddevice scikit-learn opencv-python numpy pandas
!pip install deepface
import cv2
from deepface import DeepFace
from google.colab.output import eval_js
from IPython.display import display, Javascript
import base64
import numpy as np

def video_stream():
  js = Javascript('''
    var video;
    var div = null;
    var stream;
    var captureCanvas;
    async function createDom() {
      if (div !== null) return;
      div = document.createElement('div');
      video = document.createElement('video');
      video.width = 640; video.height = 480; video.autoplay = true;
      div.appendChild(video);
      document.body.appendChild(div);
      stream = await navigator.mediaDevices.getUserMedia({video: true});
      video.srcObject = stream;
      await video.play();
    }
    createDom();
    ''')
  display(js)

# 1. Quick Verification Helper
def verify_system():
    print("Checking DeepFace and dependencies...")
    try:
        # Test DeepFace with a blank image to ensure weights are loaded
        blank_img = np.zeros((224, 224, 3), dtype=np.uint8)
        _ = DeepFace.analyze(blank_img, actions=['emotion'], enforce_detection=False)
        print("✅ DeepFace is ready.")
    except Exception as e:
        print(f"❌ DeepFace Error: {e}")

    print("\nChecking Webcam Access...")
    try:
        video_stream()
        print("✅ Webcam bridge request sent. Please allow camera access in your browser.")
    except Exception as e:
        print(f"❌ Webcam Error: {e}")

verify_system()

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# 1. Setup the Recommendation Model
data = {
    'mood': ['Happy', 'Stressed', 'Calm', 'Tired'],
    'current_workload': [2, 7, 4, 6],
    'recommended_task': ['Creative work', 'Relaxation', 'Planning', 'Simple work']
}
df = pd.DataFrame(data)
mood_map = {'Happy': 0, 'Stressed': 1, 'Calm': 2, 'Tired': 3}
df['mood_encoded'] = df['mood'].map(mood_map)

X = df[['mood_encoded', 'current_workload']]
y = df['recommended_task']
clf = DecisionTreeClassifier()
clf.fit(X, y)

def recommend_task(mood, workload):
    mood_encoded = mood_map.get(mood, 0)
    prediction = clf.predict([[mood_encoded, workload]])
    return prediction[0]

# 2. Test the function
test_scenarios = [
    ('Happy', 2),
    ('Stressed', 8),
    ('Calm', 4),
    ('Tired', 6),
    ('Happy', 9)
]

print("--- Task Recommendation Test Results ---")
for mood, workload in test_scenarios:
    recommendation = recommend_task(mood, workload)
    print(f"Mood: {mood:8} | Workload: {workload} | Recommended Task: {recommendation}")

    import pandas as pd
import numpy as np
import cv2
import base64
from datetime import datetime, timezone
from sklearn.tree import DecisionTreeClassifier
from deepface import DeepFace
from google.colab.output import eval_js
from IPython.display import display, Javascript

# --- 1. Webcam Bridge Utilities ---
def video_stream():
    js = Javascript('''
        var video; var div = null; var stream; var captureCanvas;
        var pendingResolve = null; var shutdown = false;
        function removeDom() {
            if(stream) stream.getTracks().forEach(track => track.stop());
            if(video) video.remove(); if(div) div.remove();
        }
        function onAnimationFrame() {
            if (!shutdown) window.requestAnimationFrame(onAnimationFrame);
            if (pendingResolve) {
                var result = "";
                if (!shutdown) {
                    captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
                    result = captureCanvas.toDataURL('image/jpeg', 0.8);
                }
                var resolve = pendingResolve; pendingResolve = null; resolve(result);
            }
        }
        async function createDom() {
            if (div !== null) return;
            div = document.createElement('div');
            video = document.createElement('video');
            video.width = 640; video.height = 480; video.autoplay = true;
            div.appendChild(video);
            captureCanvas = document.createElement('canvas');
            captureCanvas.width = 640; captureCanvas.height = 480;
            document.body.appendChild(div);
            stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream; await video.play();
            window.requestAnimationFrame(onAnimationFrame);
        }
        async function stream_frame(label, imgData) {
            if (shutdown) { removeDom(); shutdown = false; return "shutdown"; }
            await createDom(); return new Promise(resolve => { pendingResolve = resolve; });
        }
    ''')
    display(js)

def video_frame(label, bbox): return eval_js(f'stream_frame("{label}", "{bbox}")')
def js_to_image(js_reply):
    image_bytes = base64.b64decode(js_reply.split(',')[1])
    return cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), flags=1)

# --- 2. Emotion Logic ---
def detect_facial_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except: return "Unknown"

def hr_alert_system(emotion):
    if emotion.lower() in ['angry', 'sad', 'fear']:
        print(f"\n[ALARM] ALERT: High stress detected ({emotion.upper()}).")

# --- 3. Recommendation Model ---
data = {
    'mood': ['Happy', 'Stressed', 'Calm', 'Tired'],
    'current_workload': [2, 7, 4, 6],
    'recommended_task': ['Creative work', 'Relaxation', 'Planning', 'Simple work']
}
df_model = pd.DataFrame(data)
mood_map = {'Happy': 0, 'Stressed': 1, 'Calm': 2, 'Tired': 3}
df_model['mood_encoded'] = df_model['mood'].map(mood_map)
clf = DecisionTreeClassifier().fit(df_model[['mood_encoded', 'current_workload']], df_model['recommended_task'])

# --- 4. Execution Loop ---
mood_history_df = pd.DataFrame(columns=['timestamp', 'emotion'])
print("Starting system...")
video_stream()
count = 0
try:
    while count < 20: # Limit loop for stability in explanation
        js_reply = video_frame('Capturing...', '')
        if not js_reply or js_reply == 'shutdown': break
        img = js_to_image(js_reply)
        if count % 5 == 0:
            emotion = detect_facial_emotion(img)
            print(f"\rCurrent Detection: {emotion}", end="")
            mood_history_df.loc[len(mood_history_df)] = [datetime.now(timezone.utc), emotion]
            hr_alert_system(emotion)
        count += 1
except Exception as e: print(f"\nError: {e}")
display(mood_history_df.tail())



import pandas as pd
from datetime import datetime, timezone
from sklearn.tree import DecisionTreeClassifier

# 1. Pre-populate mood_history_df with sample data if it's empty
if 'mood_history_df' not in globals() or mood_history_df.empty:
    sample_data = [
        [datetime.now(timezone.utc), 'happy'],
        [datetime.now(timezone.utc), 'fear'],
        [datetime.now(timezone.utc), 'neutral'],
        [datetime.now(timezone.utc), 'sad']
    ]
    mood_history_df = pd.DataFrame(sample_data, columns=['timestamp', 'emotion'])
    print("Note: mood_history_df was empty. Populated with sample logs for demonstration.\n")

# 2. Define the Mapping Logic
emotion_to_mood = {
    'happy': 'Happy',
    'fear': 'Stressed',
    'angry': 'Stressed',
    'sad': 'Tired',
    'neutral': 'Calm',
    'surprise': 'Happy',
    'disgust': 'Stressed'
}

# 3. Setup and Train the Prediction Model
mood_map = {'Happy': 0, 'Stressed': 1, 'Calm': 2, 'Tired': 3}
data = {
    'mood': ['Happy', 'Stressed', 'Calm', 'Tired'],
    'current_workload': [2, 7, 4, 6],
    'recommended_task': ['Creative work', 'Relaxation', 'Planning', 'Simple work']
}
df_train = pd.DataFrame(data)
df_train['mood_encoded'] = df_train['mood'].map(mood_map)

clf = DecisionTreeClassifier()
clf.fit(df_train[['mood_encoded', 'current_workload']], df_train['recommended_task'])

# 4. Generate Predictions from History
predictions = []
simulated_workload = 5

for index, row in mood_history_df.iterrows():
    detected_emotion = row['emotion'].lower()
    mapped_mood = emotion_to_mood.get(detected_emotion, 'Calm')
    mood_encoded = mood_map[mapped_mood]

    # Use the model to predict the task
    # Note: Using [[]] to avoid sklearn feature name warnings
    task = clf.predict(pd.DataFrame([[mood_encoded, simulated_workload]], columns=['mood_encoded', 'current_workload']))[0]

    predictions.append({
        'timestamp': row['timestamp'],
        'detected': detected_emotion,
        'mapped_to': mapped_mood,
        'recommended_task': task
    })

results_df = pd.DataFrame(predictions)
print("--- Task Predictions based on Emotion History ---")
display(results_df)
