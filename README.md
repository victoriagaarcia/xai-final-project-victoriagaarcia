# Explainable AI for Driver Drowsiness Detection


This project studies the application of **Explainable Artificial Intelligence (XAI)** techniques to a computer vision model for **driver drowsiness detection**. A convolutional neural network based on a ResNet backbone is trained on facial images to classify whether a driver is drowsy or alert. Beyond predictive performance, the main objective is to **understand and critically assess the model’s decision-making process**, using a variety of local, global, and sanity-check explanation methods.

The project explores both **post-hoc explanations** (such as Occlusion Sensitivity, RISE, and Grad-CAM) and **actionable XAI**, where explanations are used to modify the input space (eye-based ROI masking) and evaluate the impact on model performance. This allows us to reason about what visual cues the model relies on, whether these cues are meaningful, and how constraining the model’s attention affects robustness and accuracy.

---

## Installation and Setup

The project is implemented in **Python 3.10** and was developed and executed on a **GPU server**. All experiments are fully reproducible using the dependency versions provided.

### 1. Recommended: create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
# On Windows use `venv\Scripts\activate`
```

### 2. Install dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains pinned versions of all libraries used in the project, including PyTorch, torchvision, OpenCV, MediaPipe, and common scientific Python packages.

This setup assumes that the server already has the appropriate CUDA drivers available for GPU execution.

### 3. CPU-only installation (local execution)
If you want to run the project locally on CPU (e.g. on a personal laptop), PyTorch must be installed using the CPU wheels. In that case, after creating and activating the virtual environment, run:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
pip install -r requirements.txt
```

## Dataset Download
The Driver Drowsiness Dataset is downloaded automatically using the provided script `download_data.py`, which relies on KaggleHub.

From the root of the project, run:
```bash
python download_data.py
```
The script will download the dataset and print the local path where the files have been stored.

**Important:** Before running the notebook, the dataset must be placed inside a folder called data, located in the same directory as the notebook.

The expected directory structure is:

```
driver_drowsiness_dataset/
├── drowsy/
│   └── *.png
└── non_drowsy/
    └── *.png
```


## Running the project
Once the environment is set up and the dataset is correctly placed, open the notebook and execute the cells in order:

1. Dataset loading and preprocessing

2. Model training (baseline CNN)

3. XAI explanations (Occlusion, RISE, Mean Grad-CAM, sanity checks)

4. Actionable XAI via eye-based ROI masking

5. Comparison between baseline and ROI-based models
