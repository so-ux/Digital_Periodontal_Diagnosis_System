import os
import nibabel as nib
import numpy as np


def reorient_to_ras(input_path, output_path):
    # Load the NIfTI file
    img = nib.load(input_path)

    # Get the image data and affine
    data = img.get_fdata()
    affine = img.affine

    # Calculate the RAS affine
    ras_affine = np.diag([1, 1, 1, 1])  # Default RAS affine
    ras_affine[:3, :3] = np.eye(3)

    # Adjust the origin
    ras_affine[:3, 3] = np.dot(affine[:3, :3], -0.5 * (np.array(data.shape[:3]) - 1)) + affine[:3, 3]

    # Create a new NIfTI image with the RAS affine
    reoriented_img = nib.Nifti1Image(data, ras_affine)

    # Save the reoriented image to the output path
    nib.save(reoriented_img, output_path)


def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            reorient_to_ras(input_path, output_path)
            print(f"Reoriented {filename} to RAS and saved to {output_path}")



# Example usage
input_directory = r'C:\Users\z\Desktop\temp4\nnunet_input'
output_directory = r'C:\Users\z\Desktop\temp4'
process_directory(input_directory, output_directory)
