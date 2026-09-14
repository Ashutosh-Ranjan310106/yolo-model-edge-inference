from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "ROD Edge Inference System - Model Registry"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Base Directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent  # ROD_Edge_Inference/backend
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    FRONTEND_DIR: Path = BASE_DIR.parent / "frontend"
    
    # External Training Directories to scan
    NAVIGATION_DIR: Path = BASE_DIR.parent.parent  # navigation root
    RUNS_DIR: Path = NAVIGATION_DIR / "runs"
    RUNS_CATALOG_FILE: Path = RUNS_DIR / "runs_catalog.json"
    TRAINING_RUNS_DIR: Path = RUNS_DIR / "train"
    
    # Supported resolutions
    SUPPORTED_RESOLUTIONS: list[int] = [320, 384, 480, 512, 640]
    
    # Fallback ROD class list (27 classes)
    ROD_CLASSES: list[str] = [
        "Bike", "Building", "Car", "Person", "Stairs",
        "Traffic sign", "Electrical Pole", "Road", "Motorcycle", "Dustbin",
        "Dog", "Manhole", "Tree", "Guard rail", "Pedestrian crosswalk",
        "Truck", "Bus", "Bench", "Traffic Cone", "Fire hydrant",
        "Teraffic Barrel", "Plant Pot", "Electrical Box", "Chair",
        "Bicycle Rack", "Door", "Wall"
    ]

settings = Settings()
