# Download Driver Drowsiness Dataset from Kaggle
import kagglehub

dataset_name = "ismailnasri20/driver-drowsiness-dataset-ddd"
path = kagglehub.dataset_download(dataset_name)
print(f"Dataset downloaded to: {path}") #.cache/kagglehub/datasets/ismailnasri20/...