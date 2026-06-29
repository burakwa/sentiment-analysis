# 🎬 Video Emotion Analysis System (Video Emotion Analyzer)

A modern, asynchronous, full-stack web application that analyzes the facial expressions of people in uploaded videos to detect their emotional states (happy, sad, angry, surprised, etc.) frame by frame.

Designed according to **Clean Architecture** principles, the project integrates image processing (MediaPipe, OpenCV) and deep learning (PyTorch/HuggingFace) models via a high-performance API with a user-friendly interface.

## ✨ Features

- 🚀 **Asynchronous Video Processing:** By processing long videos in the background (Background Tasks), HTTP timeout issues are prevented.
- 🎯 **High-Precision Face Detection:** Using Google MediaPipe, 3D landmarks of faces are extracted and mathematically aligned (Face Alignment) before the image is sent to the model.
- 🧠 **Deep Learning-Based Analysis:** Emotion classification using pre-trained Vision Transformer (ViT) or CNN models based on HuggingFace/PyTorch.
- 🎨 **Modern Interface:** A responsive and sleek user experience designed with React and Tailwind CSS, featuring drag-and-drop support.
- 🧪 **Testable Architecture:** Modular and reliable backend code supported by Pytest.

Translated with DeepL.com (free version)