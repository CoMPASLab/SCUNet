import os.path
import argparse
import gc

import numpy as np
import torch

from utils import utils_image as util
from models.network_scunet import SCUNet as net


def denoise_tiled_memory_efficient(model, img_rgb, tile_size, tile_overlap, device='cuda', use_half=False):
    """
    Memory-efficient denoising of large images by smaller overlapping tiles.

    Optimizations:
    1. Explicit memory cleanup after each tile
    2. Reduced precision for intermediate results when possible
    3. Optimized tensor operations
    4. Pre-allocated GPU tensors for tile processing

    Args:
        model: the denoising model
        img_rgb: the input noisy image (numpy array, HWC, range 0-255)
        tile_size: the size of each tile
        tile_overlap: the overlap between tiles
        device: 'cuda' or 'cpu'
        use_half: whether to use half precision (fp16)

    Returns:
        denoised_img: the denoised image (numpy array, HWC, range 0-255)
    """
    h, w, c = img_rgb.shape
    denoised_img = np.zeros_like(img_rgb, dtype=np.float32)
    weight_map = np.zeros((h, w, c), dtype=np.float32)

    # Pre-allocate tensor on GPU to avoid repeated allocations
    max_tile_h = min(tile_size, h)
    max_tile_w = min(tile_size, w)
    dtype = torch.float16 if use_half else torch.float32
    tile_tensor = torch.zeros((1, c, max_tile_h, max_tile_w),
                              dtype=dtype, device=device)

    # Process tiles with explicit memory management
    for y in range(0, h, tile_size - tile_overlap):
        for x in range(0, w, tile_size - tile_overlap):
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = x, min(x + tile_size, w)
            tile_h, tile_w = y2 - y1, x2 - x1

            # Extract tile from input image
            tile = img_rgb[y1:y2, x1:x2, :]

            # Prepare input tensor - reuse pre-allocated tensor when possible
            if tile_h == max_tile_h and tile_w == max_tile_w:
                # Reuse pre-allocated tensor
                tile_data = torch.from_numpy(tile.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
                if use_half:
                    tile_data = tile_data.half()
                tile_tensor.copy_(tile_data)
                input_tensor = tile_tensor
            else:
                # Create new tensor for different sizes (edge tiles)
                input_tensor = (torch.from_numpy(tile.astype(np.float32))
                                .permute(2, 0, 1).unsqueeze(0).to(device) / 255.0)
                if use_half:
                    input_tensor = input_tensor.half()

            # Process tile
            with torch.no_grad():
                denoised_tile = model(input_tensor)
                denoised_tile = torch.clamp(denoised_tile, 0, 1)

                # Convert back to numpy immediately and free GPU memory
                if use_half:
                    denoised_tile = denoised_tile.float()  # Convert back to float32 for numpy
                denoised_tile_np = denoised_tile.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0

                # Clear GPU tensors immediately
                del denoised_tile
                if tile_h != max_tile_h or tile_w != max_tile_w:
                    del input_tensor

                # Force GPU memory cleanup
                if device == 'cuda':
                    torch.cuda.empty_cache()

            # Accumulate results
            denoised_img[y1:y2, x1:x2, :] += denoised_tile_np
            weight_map[y1:y2, x1:x2, :] += 1.0

            # Optional: force garbage collection periodically
            if (y * w + x) % (tile_size * 4) == 0:
                gc.collect()

    # Clean up pre-allocated tensor
    del tile_tensor
    if device == 'cuda':
        torch.cuda.empty_cache()

    denoised_img /= weight_map
    return denoised_img.astype(np.uint8)


def denoise_tiled(model, img_rgb, tile_size, tile_overlap, device='cuda'):
    """
    Legacy function - redirects to memory-efficient version
    """
    return denoise_tiled_memory_efficient(model, img_rgb, tile_size, tile_overlap, device)


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='scunet_color_15',
                        help='scunet_color_15, scunet_color_25, scunet_color_50')
    parser.add_argument('--input_dir', type=str, default='testsets', help='input test image directory')
    parser.add_argument('--output_dir', type=str, default='results', help='output directory')
    parser.add_argument('--models_dir', type=str, default='models', help='path of models directory')
    parser.add_argument('--tile_size', type=int, default=1024, help='image tile size')
    parser.add_argument('--channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--half_precision', action='store_true',
                        help='use half precision (fp16) to save GPU memory')
    parser.add_argument('--cpu_mode', action='store_true',
                        help='force CPU processing (slower but uses no GPU memory)')

    args = parser.parse_args()

    # Directories
    inputs_dir = args.input_dir
    results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True)

    # Channels number
    n_channels = args.channels
    if n_channels == 1 and 'gray' not in args.model_name:
        raise ValueError('For grayscale images, please select a model with "gray" in its name.')
    if n_channels == 3 and 'gray' in args.model_name:
        raise ValueError('For color images, please select a model without "gray" in its name.')

    # Device selection with memory optimization
    if args.cpu_mode:
        device = torch.device('cpu')
        print("Using CPU mode - no GPU memory will be used")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            # Clear any existing GPU memory
            torch.cuda.empty_cache()
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"Total GPU memory: {total_memory:.1f} GB")

    # Define tiles size - adjust based on available memory
    tile_size = args.tile_size
    tile_overlap = max(1, int(tile_size * 0.05))

    # ----------------------------------------
    # load model with memory optimizations
    # ----------------------------------------
    model_path = os.path.join(args.models_dir, args.model_name+'.pth')
    model = net(in_nc=n_channels, config=[4, 4, 4, 4, 4, 4, 4], dim=64)

    # Load model weights
    checkpoint = torch.load(model_path, map_location='cpu')  # Load to CPU first
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint  # Free memory immediately

    model.eval()

    # Disable gradient computation for all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Move model to device and optionally use half precision
    model = model.to(device)
    if args.half_precision and device.type == 'cuda':
        model = model.half()
        print("Using half precision (FP16) for memory efficiency")

    print('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print('Params number: {}'.format(number_parameters))
    print('{:>16s} : {:<.4f} [M]'.format('#Params', number_parameters/10**6))

    # ----------------------------------------
    # Load and process images
    # ----------------------------------------
    path = inputs_dir
    paths = util.get_image_paths(path)

    # ----------------------------------------
    # Inference with memory management
    # ----------------------------------------
    for i, img_path in enumerate(paths):
        print(f"Processing image {i+1}/{len(paths)}: {os.path.basename(img_path)}")

        img_name, ext = os.path.splitext(os.path.basename(img_path))
        img = util.imread_uint(img_path)

        if n_channels == 3:
            # Monitor GPU memory if available
            if device.type == 'cuda':
                allocated_before = torch.cuda.memory_allocated() / 1024**3
                print(f"GPU memory before processing: {allocated_before:.2f} GB")

            # Process image
            use_half = args.half_precision and device.type == 'cuda'
            denoised_img = denoise_tiled_memory_efficient(model, img, tile_size, tile_overlap, device, use_half)

            # Monitor GPU memory after processing
            if device.type == 'cuda':
                allocated_after = torch.cuda.memory_allocated() / 1024**3
                print(f"GPU memory after processing: {allocated_after:.2f} GB")
        else:
            img = util.uint2single(img)
            img = util.single2tensor4(img)
            denoised_img = util.tensor2uint(img.to(device))

        # ------------------------------------
        # save results
        # ------------------------------------
        output_path = os.path.join(results_dir, f"{img_name}{ext}")
        util.imsave(denoised_img, output_path)
        print(f"Saved: {output_path}")

        # Force garbage collection between images
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("Processing completed!")
    if device.type == 'cuda':
        final_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Final GPU memory usage: {final_memory:.2f} GB")


if __name__ == '__main__':

    main()
