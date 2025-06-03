import os
import shutil
from datasets import Dataset, DatasetDict, Features, Image
from huggingface_hub import HfApi, create_repo

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
# (1) Point these to where you currently have the original MaSTr1325 folders:
ORIG_ROOT = "../data/MaSTr1325" 
ORIG_IMAGES = os.path.join(ORIG_ROOT, "MaSTr1325_images_512x384")
ORIG_IMUS   = os.path.join(ORIG_ROOT, "MaSTr1325_imus_512x384")
ORIG_MASKS  = os.path.join(ORIG_ROOT, "MaSTr1325_masks_512x384")

# (2) Choose a temporary “renamed” directory where we’ll copy + rename everything.
#     (Feel free to delete this folder after you’ve pushed to HF.)
RENAMED_ROOT   = "../data/MaSTr1325_renamed"
RENAMED_IMAGES = os.path.join(RENAMED_ROOT, "images_512x384")
RENAMED_IMUS   = os.path.join(RENAMED_ROOT, "imus_512x384")
RENAMED_MASKS  = os.path.join(RENAMED_ROOT, "masks_512x384")

# (3) The name of the dataset repo on HF you want to create/push to:
HF_DATASET_REPO = "Wilbur1240/MaSTr1325_512x384"  # ← change “your_hf_username” accordingly
# ────────────────────────────────────────────────────────────────────────────────


def make_dirs():
    """Create the renamed‐folders if they don’t exist, or clear them if they do."""
    for folder in (RENAMED_IMAGES, RENAMED_IMUS, RENAMED_MASKS):
        if os.path.isdir(folder):
            # Remove any leftover files (be careful!)
            for fname in os.listdir(folder):
                os.remove(os.path.join(folder, fname))
        else:
            os.makedirs(folder, exist_ok=True)


def rename_and_copy():
    """
    Iterate through the original ‘images’ folder, sort by filename,
    assign each one a 4‐digit index (0001, 0002, …),
    then copy image/imu/mask → renamed folders with new basenames.
    """
    orig_fnames = sorted([f for f in os.listdir(ORIG_IMAGES) if f.lower().endswith(".jpg")])
    total = len(orig_fnames)
    print(f"Found {total} original .jpg files in {ORIG_IMAGES}.")

    for idx, orig_fname in enumerate(orig_fnames, start=1):
        # zero-pad to 4 digits
        new_base = f"{idx:04d}"     # “0001”, “0002”, …
        # build original paths
        base_name = os.path.splitext(orig_fname)[0]  # e.g. “xyz_123”
        orig_img_path  = os.path.join(ORIG_IMAGES, base_name + ".jpg")
        orig_mask_path = os.path.join(ORIG_MASKS,  base_name + "m.png")
        if base_name.startswith("old"):
            ID = base_name.split("_")[-1]
            orig_imu_path = os.path.join(ORIG_IMUS + "/old_imu_" + ID + ".png")
        else:
            orig_imu_path  = os.path.join(ORIG_IMUS,   base_name + ".png")
        

        # sanity check: the IMU & mask must exist
        if not (os.path.isfile(orig_imu_path)):
            raise FileNotFoundError(f"Missing IMU or mask for {base_name} : {orig_imu_path}")
        if not os.path.isfile(orig_mask_path):
            raise FileNotFoundError(f"Missing IMU or mask for {base_name} : {orig_mask_path}")

        # define new paths (with zero‐padded names)
        new_img_path  = os.path.join(RENAMED_IMAGES, new_base + ".jpg")
        new_imu_path  = os.path.join(RENAMED_IMUS,   new_base + ".png")
        new_mask_path = os.path.join(RENAMED_MASKS,  new_base + ".png")

        # copy the files over under the new names
        shutil.copy2(orig_img_path,  new_img_path)
        shutil.copy2(orig_imu_path,  new_imu_path)
        shutil.copy2(orig_mask_path, new_mask_path)

        if idx % 200 == 0 or idx == total:
            print(f"  → Copied {idx}/{total}: {orig_fname} → {new_base}.(jpg/png)")


def build_and_push_hf_dataset():
    """
    1) Build a Hugging Face Dataset using the renamed folders.
    2) Push it to HF under HF_DATASET_REPO.
    """
    # 1) Gather records from the RENAMED_* folders
    records = []
    renamed_fnames = sorted([f for f in os.listdir(RENAMED_IMAGES) if f.endswith(".jpg")])
    for fname in renamed_fnames:
        base = os.path.splitext(fname)[0]  # “0001”
        img_path  = os.path.join(RENAMED_IMAGES, base + ".jpg")
        imu_path  = os.path.join(RENAMED_IMUS,   base + ".png")
        mask_path = os.path.join(RENAMED_MASKS,  base + ".png")
        # We already ensured these exist in rename_and_copy()
        records.append({
            "image": img_path,
            "imu":   imu_path,
            "mask":  mask_path
        })
    print(f"Built records for {len(records)} renamed samples.")

    # 2) Define dataset Features so HF knows these columns are images
    features = Features({
        "image": Image(),   # will store + preview the .jpg
        "imu":   Image(),   # will store + preview the .png
        "mask":  Image()    # will store + preview the .png
    })

    # 3) Build a single‐split Dataset; feel free to split into train/val/test later
    full_ds = Dataset.from_list(records, features=features)
    dataset_dict = DatasetDict({"train": full_ds})

    # 4) (Optional) peek at a couple of entries to confirm
    print("\nExample entry after renaming (should show ’0001.jpg’ etc.):")
    print(dataset_dict["train"][0])
    print(dataset_dict["train"][1])

    # 5) Create the HF dataset repo (if it doesn’t exist)
    api = HfApi(token=os.getenv("HF_TOKEN"))
    try:
        create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset")
        print(f"Created new HF dataset repo: {HF_DATASET_REPO}")
    except Exception as e:
        print(f"  • Note: could not create repo (it may already exist or you lack permissions): {e}")

    # 6) Push the DatasetDict to the Hub
    #    This will upload all  .jpg/.png files into the HF LFS storage
    dataset_dict.push_to_hub(HF_DATASET_REPO, private=False)
    print(f"\n✅ Pushed dataset to: https://huggingface.co/datasets/{HF_DATASET_REPO}")


if __name__ == "__main__":
    print("Step 1: Creating/clearing renamed‐folders …")
    make_dirs()

    print("\nStep 2: Copying + renaming every sample …")
    rename_and_copy()

    print("\nStep 3: Building HF Dataset and pushing to Hub …")
    build_and_push_hf_dataset()
