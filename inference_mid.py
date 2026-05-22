import sys
import os
from glob import glob
import argparse
from inference_pipeline_mid import InferencePipeLine
from predict_utils import ScanSegmentation

# Add current working directory to Python path
sys.path.append(os.getcwd())

# Argument parser setup
parser = argparse.ArgumentParser(description='Tooth Segmentation Inference')

parser.add_argument('--input_path', type=str, required=True, help="Path to a .obj file or directory containing .obj files.")
parser.add_argument('--save_path', type=str, required=True, help="Path to save output .json files.")

args = parser.parse_args()

# Debug: Check if paths exist
if not os.path.exists(args.input_path):
    print(f"DEBUG: Checking path: {args.input_path}")
    raise ValueError(f"Invalid input path: {args.input_path}")

# Configuration for the inference pipeline
inference_config = {
    "fps_model_info": {
        "model_parameter": {
            "input_feat": 6,
            "stride": [1, 4, 4, 4, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 512],
            "crop_sample_size": 3072,
        },
        "load_ckpt_path": "ckpts/0707_cosannealing_val"
    },
    "boundary_model_info": {
        "model_parameter": {
            "input_feat": 6,
            "stride": [1, 4, 4, 4, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 512],
            "crop_sample_size": 3072,
        },
        "load_ckpt_path": "ckpts/0711_bd_cbl_aug_test_val"
    },
    "boundary_sampling_info": {
        "bdl_ratio": 0.7,
        "num_of_bdl_points": 20000,
        "num_of_all_points": 24000,
    },
    "orginal_data_obj_path": "G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files",
}

# Collect .obj files from input path
stl_path_ls = []
if os.path.isdir(args.input_path):
    for dir_path in [x[0] for x in os.walk(args.input_path)][1:]:
        stl_path_ls += glob(os.path.join(dir_path, "*.obj"))
elif os.path.isfile(args.input_path) and args.input_path.endswith(".obj"):
    stl_path_ls.append(args.input_path)
else:
    raise ValueError(f"Invalid input path or unsupported file format: {args.input_path}")

# Initialize inference pipeline
pred_obj = ScanSegmentation(InferencePipeLine(inference_config))

# Create output directory if it doesn't exist
os.makedirs(args.save_path, exist_ok=True)

# Process each .obj file
for i, stl_path in enumerate(stl_path_ls):
    print(f"Processing {i + 1}/{len(stl_path_ls)}: {stl_path}")
    base_name = os.path.basename(stl_path).split(".")[0]
    output_file = os.path.join(args.save_path, f"{base_name}.json")
    pred_obj.process(stl_path, output_file)

print("Processing completed successfully.")
