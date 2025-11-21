# Rock–Paper–Scissors Gesture Recognition

Real-time hand-gesture classification with MediaPipe + PyTorch, supporting MQTT hardware communication.

### 1.Create Conda Environment

Create the environment using the provided environment.yml file:
```
conda env create -f environment.yml
conda activate rps
```

### 2. Detect Camera Index

If your machine has multiple cameras, run:
```
python detect_camera_index.py
```
The script will list all available camera indices.

### 3. Running the Main Program

Before running main.py, configure the following parameters inside the script:
```
BROKER_HOST = "X"
BROKER_PORT = 123456
TOPIC = "rps/gesture"
```
After setting these, change:
```
USE_MQTT = False
```
to:
```
USE_MQTT = True
```
Then run:
```
python main.py
```
All UI windows can be closed by pressing the “q” key.

### 4. Other Scripts
4.1 data_collector.py – Data Collection. Use this script to collect labeled gesture samples. Perform a gesture in front of the camera.
Press the corresponding key:
r = Rock
p = Paper
s = Scissors
Data will be automatically saved into a timestamped folder.

Run:
```
python data_collector.py
```
4.2 train_rps.py – Retraining the Model

If more data has been collected and you want to retrain the classifier:
```
python train_rps.py
```
Note: Retraining is not required for normal workflow. A pre-trained model (rps_model.pth) is already included and ready to use.
