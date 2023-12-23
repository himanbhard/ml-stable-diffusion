# from huggingface_hub import snapshot_download
# from pathlib import Path

# repo_id = "apple/coreml-stable-diffusion-v1-4"
# variant = "original/packages"

# model_path = Path("./models") / (repo_id.split("/")[-1] + "_" + variant.replace("/", "_"))
# snapshot_download(repo_id, allow_patterns=f"{variant}/*", local_dir=model_path, local_dir_use_symlinks=False)
# print(f"Model downloaded at {model_path}")

import torch
from diffusers import DiffusionPipeline

# Set the device to MPS if available, otherwise CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device {device}")

# Load the Stable Diffusion model and move it to the chosen device
pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to(device)

# Define your prompt
prompt = "An sexy woman riding a green motorcycle."

# Generate the images
generated_images = pipe(prompt=prompt)

# Access the first generated image
# Assuming the output is a list of lists, where each inner list contains image objects
first_image_list = generated_images[0]
first_image = first_image_list[0]  # Access the first image from the first list

# Save the generated image
first_image.save("generated_image1.jpg")


