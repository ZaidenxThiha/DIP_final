import os
from pathlib import Path

from ultralytics import solutions

# Ensure any generated JSON lands in this folder.
os.chdir(Path(__file__).resolve().parent)

# PARKING SPOTS SETUP
# When the popup appears, mark parking spots and save.
solutions.ParkingPtsSelection()
