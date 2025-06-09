import os
import shutil


def expand_folders(main_folder):
    """
    Moves all image files from subfolders to the main folder, renaming them if necessary.
    
    Args:
        main_folder (str): Path to the main folder where images will be moved.
    """
    if not os.path.exists(main_folder):
        raise ValueError(f"The specified folder does not exist: {main_folder}")
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    for root, dirs, files in os.walk(main_folder):
        print(f"Processing folder: {root}")
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                source = os.path.join(root, file)
                destination = os.path.join(main_folder, file)
                
                # Rename if file with same name exists
                if os.path.exists(destination):
                    base, ext = os.path.splitext(file)
                    count = 1
                    while os.path.exists(destination):
                        destination = os.path.join(main_folder, f"{base}_{count}{ext}")
                        count += 1
                
                shutil.move(source, destination)
