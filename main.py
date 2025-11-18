import cv2
import mediapipe as mp
import time

CAMERA_INDEX = 1
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
"""
this script is not trained
"""
def classify_rps(hand_landmarks, image_height, image_width):


    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]

    landmarks = hand_landmarks.landmark
    finger_states = []

    for tip_id, pip_id in zip(finger_tips, finger_pips):
        tip = landmarks[tip_id]
        pip = landmarks[pip_id]
        finger_states.append(tip.y < pip.y)

    thumb, index, middle, ring, pinky = finger_states

    straight_count = sum(finger_states)

    if straight_count <= 1:
        return "Rock"
    elif straight_count == 2 and index and middle and not ring and not pinky:
        return "Scissors"
    elif straight_count >= 4:
        return "Paper"
    else:
        return "Unknown"


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = time.time()

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

            gesture = "No hand"

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                    gesture = classify_rps(hand_landmarks, h, w)

            now = time.time()
            fps = 1.0 / (now - prev_time)
            prev_time = now

            cv2.putText(frame, f"Gesture: {gesture}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)

            cv2.imshow("RPS Gesture (Camera 1)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
