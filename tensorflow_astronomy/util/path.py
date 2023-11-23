"""Directory.
"""
import datetime
import os


def create_timestamped_directory(target_path):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    savepath = os.path.join(target_path, timestamp)
    os.makedirs(savepath, exist_ok=True)
    return savepath
