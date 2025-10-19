from ultralytics import YOLO
import argparse
import shutil
from pathlib import Path



def load_classes_from_file(class_file: str) -> list:
    """Load class names from a text file, one class name per line."""
    with open(class_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO model with custom classes file")
    parser.add_argument(
        "--classes_file",
        "-c",
        default="custom_classes.txt",
        help="Path to custom classes file (one class per line). Defaults to 'custom_classes.txt'.",
    )
    args = parser.parse_args()

    # Initialize a YOLO-World model
    model = YOLO("./src/Code_YOLO/yolov8s-world.pt")  # or select yolov8m/l-world.pt
    # Train the model on the COCO8 example dataset for 100 epochs
    data_file = "./src/Code_YOLO/Grocery1.v1i.yolov8/data.yaml"  # path to dataset config file
    results = model.train(data=data_file, epochs=100, imgsz=640)
    # Define custom classes
    classes = load_classes_from_file(args.classes_file)  # e.g., ["person", "bus"]
    #model.set_classes(classes)

    # Save the model with the defined offline vocabulary
    model.save("./src/Code_YOLO/custom_yolov8s.pt")
    print(results)