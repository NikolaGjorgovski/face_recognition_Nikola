import os
import glob
import numpy as np
from utils import encode_face, load_encodings, save_encodings
from config import DATASET_DIR, ENCODINGS_FILE


def process_dataset() -> tuple:

    encodings = []
    names = []
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory not found: {DATASET_DIR}")
        print("Creating dataset directory structure...")
        os.makedirs(DATASET_DIR, exist_ok=True)
        print(f"\nPlease organize your images in the following structure:")
        print(f"{DATASET_DIR}/")
        print("  person_name1/")
        print("    image1.jpg")
        print("    image2.jpg")
        print("  person_name2/")
        print("    image1.jpg")
        print("    ...")
        return [], []
    
    person_dirs = [d for d in os.listdir(DATASET_DIR)
                   if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    if len(person_dirs) == 0:
        print(f"No person directories found in {DATASET_DIR}")
        print("Please create subdirectories named after each person and add their images.")
        return [], []
    
    print(f"Found {len(person_dirs)} person(s) in dataset")
    print("-" * 50)
    
    for person_name in person_dirs:
        person_path = os.path.join(DATASET_DIR, person_name)
        print(f"\nProcessing: {person_name}")
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(person_path, ext)))
            image_files.extend(glob.glob(os.path.join(person_path, ext.upper())))
        
        if len(image_files) == 0:
            print(f"  No images found in {person_name}/")
            continue
        
        print(f"  Found {len(image_files)} image(s)")
        
        processed_count = 0
        for image_path in image_files:
            print(f"  Encoding: {os.path.basename(image_path)}", end=" ... ")
            
            face_encoding = encode_face(image_path)
            
            if face_encoding is not None:
                encodings.append(face_encoding)
                names.append(person_name)
                processed_count += 1
                print("✓")
            else:
                print("✗ (no face detected)")
        
        print(f"  Successfully processed {processed_count}/{len(image_files)} images")
    
    print("-" * 50)
    print(f"\nTotal encodings generated: {len(encodings)}")
    
    return encodings, names


def main():

    print("=" * 50)
    print("Face Recognition Training Script")
    print("=" * 50)
    
    existing_encodings, existing_names = load_encodings()
    
    new_encodings, new_names = process_dataset()
    
    if len(new_encodings) == 0:
        print("\nNo face encodings were generated. Please check your dataset.")
        return
    
    if len(existing_encodings) > 0:
        print(f"\nMerging with {len(existing_encodings)} existing encodings...")
        all_encodings = existing_encodings + new_encodings
        all_names = existing_names + new_names
    else:
        all_encodings = new_encodings
        all_names = new_names
    
    if save_encodings(all_encodings, all_names):
        print(f"\n✓ Training complete! {len(all_encodings)} total encodings saved.")
        print(f"  You can now run 'python recognize.py' for real-time recognition.")
    else:
        print("\n✗ Failed to save encodings.")


if __name__ == "__main__":
    main()




