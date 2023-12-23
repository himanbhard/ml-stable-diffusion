from huggingface_hub import snapshot_download
from pathlib import Path
import python_coreml_stable_diffusion.pipeline
import os

output_dir = "./output"

# Check if the directory does not exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    print(f"Output directory created at {output_dir}")
else:
    print(f"Output directory already exists at {output_dir}")

repo_id = "apple/coreml-stable-diffusion-2-1-base"
variant = "original/packages"


model_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))

# Check if the model files already exist
if not model_path.exists():
    snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
    print(f"Model downloaded at {model_path}")
else:
    print(f"Model already exists at {model_path}")



# python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i mlpackages -o output --compute-unit ALL --seed 93 --model-version CompVis/stable-diffusion-v1-4
# python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i mlpackages -o output  --compute-unit ALL --seed 93 --model-version stabilityai/stable-diffusion-2-1-base




# command to use stable diffusion 2.1 base
# python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i mlpackages -o output --compute-unit ALL --seed 93 --model-version apple/coreml-stable-diffusion-2-1-base
#python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i mlpackages -o output --seed 93 -i models/coreml-stable-diffusion-2-base_original_packages --model-version stabilityai/stable-diffusion-2-base
