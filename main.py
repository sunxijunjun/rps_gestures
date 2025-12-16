import cv2
import mediapipe as mp
import time
import numpy as np
import torch
import torch.nn as nn
import json

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None
    print("[WARN] paho-mqtt not installed. USE_MQTT must be False!")

CAMERA_INDEX = 0
MODEL_PATH = "rps_model.pth"

USE_MQTT = True
BROKER_HOST = "broker.hivemq.com"
BROKER_PORT = 1883
TOPIC = "rps/gesture"

STABLE_FRAMES = 10

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

class MLP(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_trained_model(model_path: str, device):
    checkpoint = torch.load(model_path, map_location=device)

    input_dim = checkpoint["input_dim"]
    label_map = checkpoint["label_map"]  # {"rock":0, "paper":1, "scissors":2}
    num_classes = len(label_map)

    model = MLP(input_dim=input_dim, hidden_dim=128, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    inv_label_map = {v: k for k, v in label_map.items()}

    return model, inv_label_map, input_dim


def landmarks_to_features(hand_landmarks) -> np.ndarray:
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.asarray(data, dtype=np.float32)


def classify_rps_with_model(
    hand_landmarks,
    image_height,
    image_width,
    model,
    device,
    inv_label_map,
    input_dim,
):
    feat = landmarks_to_features(hand_landmarks)  # shape: (63,)

    if feat.shape[0] != input_dim:
        return "Unknown"

    x = torch.from_numpy(feat).unsqueeze(0).to(device)  # (1, 63)

    with torch.no_grad():
        logits = model(x)
        pred_idx = torch.argmax(logits, dim=1).item()

    label = inv_label_map.get(pred_idx, "Unknown")

    return label.capitalize()  # "rock" -> "Rock"


# ========= MQTT / Dummy MQTT =========

class DummyMQTTClient:
    def publish(self, topic, payload, qos=0):
        print(f"[DummyMQTT] topic={topic}, qos={qos}, payload={payload}")

def setup_mqtt_client():
    if not USE_MQTT:
        print("[INFO] USE_MQTT = False, using DummyMQTTClient (no real network).")
        return DummyMQTTClient()

    if mqtt is None:
        raise RuntimeError(
            "paho-mqtt not installed. 请先 `pip install paho-mqtt`，或将 USE_MQTT 设为 False。"
        )

    client = mqtt.Client()
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    print(f"[INFO] Connected to MQTT broker at {BROKER_HOST}:{BROKER_PORT}")
    return client


def send_gesture_mqtt(mqtt_client, gesture: str):
    payload = {"gesture": gesture}
    msg = json.dumps(payload)
    mqtt_client.publish(TOPIC, msg, qos=0)
    print("[MQTT] Sent:", msg)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, inv_label_map, input_dim = load_trained_model(MODEL_PATH, device)
    print(f"Loaded model from {MODEL_PATH}, input_dim={input_dim}")

    # MQTT / DummyMQTT
    mqtt_client = setup_mqtt_client()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()

    prev_gesture_raw = None      # Last user gesture
    same_count = 0               # same for X frames
    last_sent_gesture = None     # last sent MQTT gesture
    stable_gesture = "None"      # current stable gesture

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # BGR -> RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            gesture_raw = "No hand"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    gesture_raw = classify_rps_with_model(
                        hand_landmarks,
                        h,
                        w,
                        model,
                        device,
                        inv_label_map,
                        input_dim,
                    )

            if gesture_raw == prev_gesture_raw:
                same_count += 1
            else:
                prev_gesture_raw = gesture_raw
                same_count = 1

            if (
                same_count >= STABLE_FRAMES
                and gesture_raw not in ["No hand", "Unknown"]
                and gesture_raw != last_sent_gesture
            ):
                stable_gesture = gesture_raw
                last_sent_gesture = gesture_raw
                send_gesture_mqtt(mqtt_client, stable_gesture)

            now = time.time()
            fps = 1.0 / (now - prev_time)
            prev_time = now

            cv2.putText(frame, f"Raw: {gesture_raw}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)
            cv2.putText(frame, f"Stable: {stable_gesture}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)

            cv2.imshow("RPS Gesture (Trained Model + MQTT)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
