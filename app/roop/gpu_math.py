import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# Initialize device once
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if str(device) == 'cuda':
    logger.info("ROOP_GPU_MATH: CUDA is available. Using PyTorch for pre/post processing math.")
else:
    logger.info("ROOP_GPU_MATH: CUDA is not available. Falling back to OpenCV CPU processing.")

def _get_theta(M, src_shape, dst_shape):
    """Converts OpenCV affine matrix M to PyTorch affine_grid theta"""
    M_inv = cv2.invertAffineTransform(M)
    M_inv_3x3 = np.vstack([M_inv, [0, 0, 1]])
    
    H_s, W_s = src_shape
    H_d, W_d = dst_shape
    
    T_dst = np.array([
        [W_d / 2.0, 0, W_d / 2.0 - 0.5],
        [0, H_d / 2.0, H_d / 2.0 - 0.5],
        [0, 0, 1]
    ], dtype=np.float32)
    
    T_src_inv = np.array([
        [2.0 / W_s, 0, 1.0 / W_s - 1.0],
        [0, 2.0 / H_s, 1.0 / H_s - 1.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    theta_3x3 = T_src_inv @ M_inv_3x3 @ T_dst
    return theta_3x3[:2, :]

def warp_affine_cuda(src_np, M, dsize, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0, flags=cv2.INTER_LINEAR):
    """PyTorch CUDA equivalent for cv2.warpAffine"""
    if str(device) == 'cpu':
        return cv2.warpAffine(src_np, M, dsize, flags=flags, borderMode=borderMode, borderValue=borderValue)
    
    W_d, H_d = dsize
    dst_shape = (H_d, W_d)
    
    is_2d = len(src_np.shape) == 2
    if is_2d:
        src_np = src_np[:, :, np.newaxis]
        
    src_shape = src_np.shape[:2]
    
    # Math happens here
    with torch.no_grad():
        theta = _get_theta(M, src_shape, dst_shape)
        theta_tensor = torch.from_numpy(theta).unsqueeze(0).to(device, dtype=torch.float32)
        
        # Normalize input to float tensor
        is_uint8 = src_np.dtype == np.uint8
        
        src_tensor = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)
        
        padding_mode = 'zeros' if borderMode == cv2.BORDER_CONSTANT else 'border'
        
        grid = F.affine_grid(theta_tensor, (1, src_tensor.shape[1], H_d, W_d), align_corners=False)
        dst_tensor = F.grid_sample(src_tensor, grid, mode='bilinear', padding_mode=padding_mode, align_corners=False)
        
        dst_np = dst_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        if is_uint8:
            dst_np = np.clip(dst_np, 0, 255).astype(np.uint8)
            
        if is_2d:
            dst_np = dst_np[:, :, 0]
            
        return dst_np

def _get_gaussian_kernel(kernel_size, sigma):
    """Generates a 2D Gaussian kernel"""
    if sigma <= 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    x = np.arange(kernel_size) - kernel_size // 2
    x = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d = x / x.sum()
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d

def gaussian_blur_cuda(src_np, ksize, sigmaX=0):
    """PyTorch CUDA equivalent for cv2.GaussianBlur"""
    if str(device) == 'cpu':
        return cv2.GaussianBlur(src_np, ksize, sigmaX)
    
    kernel_size = ksize[0]
    if kernel_size <= 1:
        return src_np
        
    is_2d = len(src_np.shape) == 2
    if is_2d:
        src_np = src_np[:, :, np.newaxis]
        
    channels = src_np.shape[2]
    
    with torch.no_grad():
        kernel_2d = _get_gaussian_kernel(kernel_size, sigmaX)
        kernel_tensor = torch.from_numpy(kernel_2d).to(device, dtype=torch.float32)
        kernel_tensor = kernel_tensor.view(1, 1, kernel_size, kernel_size)
        kernel_tensor = kernel_tensor.repeat(channels, 1, 1, 1)
        
        is_uint8 = src_np.dtype == np.uint8
        
        src_tensor = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)
        
        padding = kernel_size // 2
        # OpenCV replicates border for GaussianBlur by default
        src_tensor = F.pad(src_tensor, (padding, padding, padding, padding), mode='replicate')
        dst_tensor = F.conv2d(src_tensor, kernel_tensor, groups=channels)
        
        dst_np = dst_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        if is_uint8:
            dst_np = np.clip(dst_np, 0, 255).astype(np.uint8)
            
        if is_2d:
            dst_np = dst_np[:, :, 0]
            
        return dst_np

def add_weighted_cuda(src1, alpha, src2, beta, gamma):
    """PyTorch CUDA equivalent for cv2.addWeighted"""
    if str(device) == 'cpu':
        return cv2.addWeighted(src1, alpha, src2, beta, gamma)
        
    with torch.no_grad():
        t1 = torch.from_numpy(src1).to(device, dtype=torch.float32)
        t2 = torch.from_numpy(src2).to(device, dtype=torch.float32)
        
        dst = t1 * alpha + t2 * beta + gamma
        
        if src1.dtype == np.uint8:
            dst = torch.clamp(dst, 0, 255)
            
        return dst.cpu().numpy().astype(src1.dtype)
