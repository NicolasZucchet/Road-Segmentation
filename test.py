from src.images import load_train_data_directories
import os 

from src.data import RoadSegmentationDataset

RoadSegmentationDataset('./data/CIL/training', subtasks=False, indices=slice(70), train=True)

"""
import os
for p in os.walk("data/GoogleMaps"):
    if 'labels' in p[0]:
        new_name = "/".join(p[0].split("/")[:-1]) + "/groundtruth"
        os.rename(p[0], new_name)
print("HA")
"""

