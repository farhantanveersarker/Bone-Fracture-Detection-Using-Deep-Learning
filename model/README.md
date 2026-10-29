# Model

Store trained model weights, checkpoints, and helper scripts here.

- `preprocess_data.py`: builds TensorFlow image generators.
- `train_mobilenetv2.py`: trains a fracture detection model using MobileNetV2 transfer learning.
- `evaluate_model.py`: evaluates the saved model on the val/test sets and writes `metrics.json`.
