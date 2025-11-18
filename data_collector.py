import cv2
import mediapipe as mp
import csv
import os
from datetime import datetime

CAMERA_INDEX = 1 #Check if this is the right index!!!!!!!!

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
CSV_PATH = os.path.join(DATA_DIR, f"rps_data_{timestamp}.csv")


def init_csv(path):
    file_exists = os.path.exists(path)
    if not file_exists:
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["label"]
            for i in range(21):
                header += [f"x_{i}", f"y_{i}", f"z_{i}"]
            writer.writerow(header)


def save_sample(label, hand_landmarks):
    row = [label]
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])

    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main():
    init_csv(CSV_PATH)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    counts = {"Rock": 0, "Paper": 0, "Scissors": 0}
    last_info = ""

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            current_hand = None

            if results.multi_hand_landmarks:
                current_hand = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, current_hand, mp_hands.HAND_CONNECTIONS
                )

            cv2.putText(frame,
                        f"Saving to: {os.path.basename(CSV_PATH)}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)

            cv2.putText(
                frame,
                "Press r: Rock, p: Paper, s: Scissors, q: Quit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Counts - Rock: {counts['Rock']}, Paper: {counts['Paper']}, Scissors: {counts['Scissors']}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            if last_info:
                cv2.putText(
                    frame,
                    last_info,
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("RPS Data Collection (Camera 1)", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if current_hand is None:
                if key in (ord("r"), ord("p"), ord("s")):
                    last_info = "No hand detected - sample NOT saved."
                continue

            if key == ord("r"):
                save_sample("rock", current_hand)
                counts["Rock"] += 1
                last_info = f"Saved ROCK"
            elif key == ord("p"):
                save_sample("paper", current_hand)
                counts["Paper"] += 1
                last_info = f"Saved PAPER"
            elif key == ord("s"):
                save_sample("scissors", current_hand)
                counts["Scissors"] += 1
                last_info = f"Saved SCISSORS"

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
