#!/usr/bin/env python3
"""
CLI Demo for KV-Edit: Training-Free Image Editing for Precise Background Preservation

This script provides a command-line interface for the KV-Edit image editing system.
It allows you to edit images by specifying source and target prompts along with various parameters.

Usage:
    python cli_kv_edit.py --input_image path/to/image.jpg --mask_image path/to/mask.png \
                          --source_prompt "original description" --target_prompt "desired edit" \
                          [additional options...]

Arguments:
    --input_image: Path to the input image that needs to be edited
    --mask_image: Path to the mask image (white areas will be edited, black areas preserved)
    --source_prompt: Text description of the original image content
    --target_prompt: Text description of the desired edited result
    --output_dir: Directory to save the results (default: 'regress_result')
    --width: Image width, will be adjusted to multiple of 16 (default: auto-detect from image)
    --height: Image height, will be adjusted to multiple of 16 (default: auto-detect from image)
    --inversion_num_steps: Number of steps for DDIM inversion (default: 28)
                          Higher values = more accurate inversion but slower
    --denoise_num_steps: Number of steps for denoising process (default: 28)
                        Higher values = better quality but slower
    --skip_step: Number of steps to skip during editing (default: 4)
                Lower values = more faithful to target prompt but may affect background preservation
    --inversion_guidance: Guidance scale for inversion process (default: 1.5)
                         Higher values = stronger adherence to source prompt during inversion
    --denoise_guidance: Guidance scale for denoising process (default: 5.5)
                       Higher values = stronger adherence to target prompt during editing
    --attn_scale: Attention scale between mask and background (default: 1.0)
                 Higher values = better preservation of background details
    --seed: Random seed for reproducibility (default: 42, use -1 for random)
    --re_init: Enable re-initialization for better editing quality (may affect background)
    --attn_mask: Enable attention masking for enhanced editing performance
    --name: Model name to use (default: flux-dev, options: flux-dev, flux-schnell)
    --device: Device to use for computation (default: cuda if available, else cpu)
    --gpus: Use two GPUs for distributed processing
"""

import os
import re
import time
import argparse
from dataclasses import dataclass
from glob import iglob
from einops import rearrange
from PIL import ExifTags, Image
import torch
import numpy as np
from flux.sampling import prepare
from flux.util import (configs, load_ae, load_clip, load_t5)
from models.kv_edit import Flux_kv_edit


@dataclass
class SamplingOptions:
    source_prompt: str = ''
    target_prompt: str = ''
    width: int = 1366
    height: int = 768
    inversion_num_steps: int = 0
    denoise_num_steps: int = 0
    skip_step: int = 0
    inversion_guidance: float = 1.0
    denoise_guidance: float = 1.0
    seed: int = 42
    re_init: bool = False
    attn_mask: bool = False
    attn_scale: float = 1.0


class FluxEditor_CLI:
    def __init__(self, args):
        """
        Initialize the Flux Editor CLI
        
        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.gpus = args.gpus
        if self.gpus:
            self.device = [torch.device("cuda:0"), torch.device("cuda:1")]
        else:
            self.device = [torch.device(args.device), torch.device(args.device)]

        self.name = args.name
        self.is_schnell = args.name == "flux-schnell"
        self.output_dir = args.output_dir

        print(f"Loading models on devices: {self.device}")
        print("Loading T5 text encoder...")
        self.t5 = load_t5(self.device[1], max_length=256 if self.name == "flux-schnell" else 512)
        
        print("Loading CLIP text encoder...")
        self.clip = load_clip(self.device[1])
        
        print("Loading Flux KV-Edit model...")
        self.model = Flux_kv_edit(self.device[0], name=self.name)
        
        print("Loading autoencoder...")
        self.ae = load_ae(self.name, device=self.device[1])

        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()
        
        self.info = {}
        print("All models loaded successfully!")

    def load_and_prepare_images(self, input_image_path, mask_image_path):
        """
        Load and prepare input image and mask
        
        Args:
            input_image_path: Path to input image
            mask_image_path: Path to mask image
            
        Returns:
            tuple: (processed_image, mask_tensor, height, width)
        """
        # Load input image
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"Input image not found: {input_image_path}")
        
        init_image = Image.open(input_image_path).convert('RGB')
        init_image = np.array(init_image)
        
        # Load mask image
        if not os.path.exists(mask_image_path):
            raise FileNotFoundError(f"Mask image not found: {mask_image_path}")
        
        mask_image = Image.open(mask_image_path).convert('L')  # Convert to grayscale
        mask_image = np.array(mask_image)
        
        # Ensure mask and image have same dimensions
        if mask_image.shape[:2] != init_image.shape[:2]:
            mask_image = Image.fromarray(mask_image).resize((init_image.shape[1], init_image.shape[0]))
            mask_image = np.array(mask_image)
        
        # Adjust dimensions to be multiple of 16
        height = init_image.shape[0] if init_image.shape[0] % 16 == 0 else init_image.shape[0] - init_image.shape[0] % 16
        width = init_image.shape[1] if init_image.shape[1] % 16 == 0 else init_image.shape[1] - init_image.shape[1] % 16
        
        init_image = init_image[:height, :width, :]
        mask_image = mask_image[:height, :width]
        
        # Normalize mask to 0-1 range
        mask = mask_image.astype(float) / 255.0
        mask = mask.astype(int)  # Convert to binary mask
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device[0])
        
        return init_image, mask, height, width

    @torch.inference_mode()
    def encode(self, init_image, torch_device):
        """
        Encode image using the autoencoder
        
        Args:
            init_image: Input image as numpy array
            torch_device: Device to use for encoding
            
        Returns:
            torch.Tensor: Encoded image
        """
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
        init_image = init_image.unsqueeze(0) 
        init_image = init_image.to(torch_device)
        self.ae.encoder.to(torch_device)
        
        init_image = self.ae.encode(init_image).to(torch.bfloat16)
        return init_image

    @torch.inference_mode()
    def inverse(self, init_image, mask, opts):
        """
        Perform DDIM inversion on the input image
        
        Args:
            init_image: Encoded input image
            mask: Mask tensor
            opts: Sampling options
            
        Returns:
            tuple: (z0, zt, info) - inverted latents and inversion info
        """
        print("Starting image inversion...")
        
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        torch.cuda.empty_cache()
        
        t0 = time.perf_counter()
        
        with torch.no_grad():
            inp = prepare(self.t5, self.clip, init_image, prompt=opts.source_prompt)
            z0, zt, info = self.model.inverse(inp, mask, opts)
            
        t1 = time.perf_counter()
        print(f"Inversion completed in {t1 - t0:.1f}s")
        
        return z0, zt, info

    @torch.inference_mode()
    def edit(self, z0, zt, info, init_image, mask, opts):
        """
        Perform image editing using the inverted latents
        
        Args:
            z0: Initial latent
            zt: Inverted latent
            info: Inversion information
            init_image: Original encoded image
            mask: Mask tensor
            opts: Sampling options
            
        Returns:
            PIL.Image: Edited image
        """
        print("Starting image editing...")
        
        torch.cuda.empty_cache()
        
        # Set random seed
        seed = opts.seed
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
            
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        t0 = time.perf_counter()

        with torch.no_grad():
            inp_target = prepare(self.t5, self.clip, init_image, prompt=opts.target_prompt)
            x = self.model.denoise(z0.clone(), zt, inp_target, mask, opts, info)
            
        with torch.autocast(device_type=self.device[1].type, dtype=torch.bfloat16):
            x = self.ae.decode(x.to(self.device[1]))
    
        x = x.clamp(-1, 1)
        x = x.float().cpu()
        x = rearrange(x[0], "c h w -> h w c")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t1 = time.perf_counter()
        print(f"Editing completed in {t1 - t0:.1f}s")
        
        return x

    def save_result(self, edited_image, opts, mask_path):
        """
        Save the edited image with metadata
        
        Args:
            edited_image: Edited image tensor
            opts: Sampling options
            mask_path: Path to mask image (for copying)
            
        Returns:
            str: Path to saved image
        """
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Generate output filename
        output_name = os.path.join(self.output_dir, "img_{idx}.jpg")
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0
        
        fn = output_name.format(idx=idx)
        
        # Convert tensor to PIL image
        img = Image.fromarray((127.5 * (edited_image + 1.0)).cpu().byte().numpy())
        
        # Add EXIF metadata
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = self.name
        exif_data[ExifTags.Base.ImageDescription] = opts.target_prompt
        
        # Save image
        img.save(fn, exif=exif_data, quality=95, subsampling=0)
        
        # Copy mask to output directory
        mask_output = fn.replace(".jpg", "_mask.png")
        Image.open(mask_path).save(mask_output)
        
        print(f"Results saved:")
        print(f"  Edited image: {fn}")
        print(f"  Mask: {mask_output}")
        
        return fn

    def run(self):
        """
        Main execution function
        """
        try:
            # Load and prepare images
            print("Loading images...")
            init_image, mask, height, width = self.load_and_prepare_images(
                self.args.input_image, self.args.mask_image
            )
            
            # Override width/height if specified
            if self.args.width > 0:
                width = self.args.width
            if self.args.height > 0:
                height = self.args.height
            
            # Create sampling options
            opts = SamplingOptions(
                source_prompt=self.args.source_prompt,
                target_prompt=self.args.target_prompt,
                width=width,
                height=height,
                inversion_num_steps=self.args.inversion_num_steps,
                denoise_num_steps=self.args.denoise_num_steps,
                skip_step=self.args.skip_step,
                inversion_guidance=self.args.inversion_guidance,
                denoise_guidance=self.args.denoise_guidance,
                seed=self.args.seed,
                re_init=self.args.re_init,
                attn_mask=self.args.attn_mask,
                attn_scale=self.args.attn_scale
            )
            
            print(f"Image dimensions: {width}x{height}")
            print(f"Source prompt: '{opts.source_prompt}'")
            print(f"Target prompt: '{opts.target_prompt}'")
            
            # Encode input image
            encoded_image = self.encode(init_image, self.device[1]).to(self.device[0])
            
            # Perform inversion
            z0, zt, info = self.inverse(encoded_image, mask if opts.attn_mask else None, opts)
            
            # Perform editing
            edited_image = self.edit(z0, zt, info, encoded_image, mask, opts)
            
            # Save results
            output_path = self.save_result(edited_image, opts, self.args.mask_image)
            
            print(f"\n✅ Editing completed successfully!")
            print(f"Output saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error during processing: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="CLI Demo for KV-Edit: Training-Free Image Editing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--input_image", 
        type=str, 
        required=True,
        help="Path to the input image that needs to be edited"
    )
    parser.add_argument(
        "--mask_image", 
        type=str, 
        required=True,
        help="Path to the mask image (white/bright areas will be edited, black/dark areas preserved)"
    )
    parser.add_argument(
        "--source_prompt", 
        type=str, 
        required=True,
        help="Text description of the original image content"
    )
    parser.add_argument(
        "--target_prompt", 
        type=str, 
        required=True,
        help="Text description of the desired edited result"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="regress_result",
        help="Directory to save the results (default: regress_result)"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=-1,
        help="Image width, will be adjusted to multiple of 16 (default: auto-detect from image)"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=-1,
        help="Image height, will be adjusted to multiple of 16 (default: auto-detect from image)"
    )
    parser.add_argument(
        "--inversion_num_steps", 
        type=int, 
        default=28,
        help="Number of steps for DDIM inversion (default: 28). Higher values = more accurate inversion but slower"
    )
    parser.add_argument(
        "--denoise_num_steps", 
        type=int, 
        default=28,
        help="Number of steps for denoising process (default: 28). Higher values = better quality but slower"
    )
    parser.add_argument(
        "--skip_step", 
        type=int, 
        default=4,
        help="Number of steps to skip during editing (default: 4). Lower values = more faithful to target prompt but may affect background preservation"
    )
    parser.add_argument(
        "--inversion_guidance", 
        type=float, 
        default=1.5,
        help="Guidance scale for inversion process (default: 1.5). Higher values = stronger adherence to source prompt during inversion"
    )
    parser.add_argument(
        "--denoise_guidance", 
        type=float, 
        default=5.5,
        help="Guidance scale for denoising process (default: 5.5). Higher values = stronger adherence to target prompt during editing"
    )
    parser.add_argument(
        "--attn_scale", 
        type=float, 
        default=1.0,
        help="Attention scale between mask and background (default: 1.0). Higher values = better preservation of background details"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42). Use -1 for random seed"
    )
    parser.add_argument(
        "--re_init", 
        action="store_true",
        help="Enable re-initialization for better editing quality (may affect background preservation)"
    )
    parser.add_argument(
        "--attn_mask", 
        action="store_true",
        help="Enable attention masking for enhanced editing performance"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="flux-dev", 
        choices=list(configs.keys()),
        help="Model name to use (default: flux-dev)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--gpus", 
        action="store_true",
        help="Use two GPUs for distributed processing"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_image):
        print(f"❌ Error: Input image not found: {args.input_image}")
        return
    
    if not os.path.exists(args.mask_image):
        print(f"❌ Error: Mask image not found: {args.mask_image}")
        return
    
    if not args.source_prompt.strip():
        print("❌ Error: Source prompt cannot be empty")
        return
    
    if not args.target_prompt.strip():
        print("❌ Error: Target prompt cannot be empty")
        return
    
    # Create and run editor
    editor = FluxEditor_CLI(args)
    editor.run()


if __name__ == "__main__":
    main() 