import os
from pathlib import Path

from ultralytics import solutions

# Make sure any generated `bounding_boxes.json` is saved in this folder.
os.chdir(Path(__file__).resolve().parent)

# PARKING SPOTS SETUP
# When the popup appears, mark parking bays and save.
solutions.ParkingPtsSelection()
