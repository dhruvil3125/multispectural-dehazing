# Satellite Image Dehazing Troubleshooting Guide

## Why Your Dehazed Satellite Image Is Not Visible

This comprehensive guide addresses the most common issues when satellite image dehazing results are not visible or appear incorrect.

## 🔍 Quick Diagnosis Checklist

### 1. **Data Type Issues** (Most Common Cause)
- **Problem**: Output image has wrong data type for display
- **Symptoms**: Black image, white image, no visible content
- **Quick Fix**:
  ```python
  # If your dehazed result is float
  dehazed_visible = (np.clip(dehazed_result, 0, 1) * 255).astype(np.uint8)
  
  # If values are outside [0,1] range
  dehazed_normalized = (dehazed_result - dehazed_result.min()) / (dehazed_result.max() - dehazed_result.min())
  dehazed_visible = (dehazed_normalized * 255).astype(np.uint8)
  ```

### 2. **Value Range Problems**
- **Problem**: Image values outside displayable range
- **Symptoms**: All black, all white, extreme contrast
- **Solutions**:
  ```python
  # Method 1: Normalize to [0,255]
  cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  
  # Method 2: Manual clipping
  np.clip(image, 0, 255).astype(np.uint8)
  
  # Method 3: Rescale intensity
  from skimage import exposure
  exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
  ```

### 3. **Insufficient Contrast Enhancement**
- **Problem**: Dehazing algorithm worked but improvement is subtle
- **Symptoms**: Minimal visible difference from original
- **Solutions**:
  ```python
  # Apply additional contrast enhancement
  enhanced = cv2.equalizeHist(dehazed_image.astype(np.uint8))
  
  # Or use CLAHE
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  enhanced = clahe.apply(dehazed_image.astype(np.uint8))
  
  # Gamma correction
  gamma_corrected = np.power(dehazed_image / 255.0, 0.6) * 255
  ```

### 4. **Algorithm Parameter Issues**
- **Problem**: Dehazing parameters not suitable for your image
- **Common Issues**:
  - Transmission map too conservative
  - Wrong atmospheric light estimation
  - Inappropriate patch size for image resolution
- **Solutions**: See parameter tuning section below

## 🛠️ Complete Working Solutions

### Method 1: Dark Channel Prior (Recommended)
```python
def improved_dark_channel_dehazing(image):
    # Ensure float32 [0,1] range
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = np.clip(image.astype(np.float32), 0, 1)
    
    # Parameters for better results
    patch_size = 15
    omega = 0.95
    t0 = 0.1
    
    # Dark channel calculation
    dark_channel = np.min(img, axis=2)
    kernel = np.ones((patch_size, patch_size))
    dark_channel = cv2.erode(dark_channel, kernel)
    
    # Atmospheric light estimation (top 0.1% brightest pixels)
    flat_dark = dark_channel.flatten()
    flat_img = img.reshape(-1, 3)
    indices = np.argpartition(flat_dark, -int(0.001 * len(flat_dark)))[-int(0.001 * len(flat_dark)):]
    atmospheric_light = np.max(flat_img[indices], axis=0)
    
    # Transmission estimation
    transmission = 1 - omega * np.min(img / atmospheric_light, axis=2)
    transmission = np.maximum(transmission, t0)
    
    # Scene recovery
    result = np.zeros_like(img)
    for i in range(3):
        result[:,:,i] = (img[:,:,i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    
    # CRITICAL: Proper output formatting
    result_clipped = np.clip(result, 0, 1)
    result_uint8 = (result_clipped * 255).astype(np.uint8)
    
    return result_uint8
```

### Method 2: Contrast Enhancement Approach
```python
def contrast_based_dehazing(image):
    # Convert to proper format
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = np.clip(image, 0, 1).astype(np.float32)
    
    # Multi-step enhancement
    # Step 1: Gamma correction
    gamma_corrected = np.power(img, 0.6)
    
    # Step 2: Histogram equalization in LAB space
    from skimage.color import rgb2lab, lab2rgb
    from skimage import exposure
    
    lab = rgb2lab(gamma_corrected)
    lab[:,:,0] = exposure.equalize_adapthist(lab[:,:,0], clip_limit=0.03)
    enhanced = lab2rgb(lab)
    
    # Step 3: Final contrast adjustment
    final = exposure.rescale_intensity(enhanced, out_range=(0, 1))
    
    return (np.clip(final, 0, 1) * 255).astype(np.uint8)
```

### Method 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
```python
def clahe_dehazing(image):
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            img_uint8 = (image * 255).astype(np.uint8)
        else:
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        img_uint8 = image.copy()
    
    # Apply CLAHE to each channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    
    if len(img_uint8.shape) == 3:
        result = np.zeros_like(img_uint8)
        for i in range(3):
            result[:,:,i] = clahe.apply(img_uint8[:,:,i])
    else:
        result = clahe.apply(img_uint8)
    
    return result
```

## 🔧 Parameter Tuning Guide

### Dark Channel Prior Parameters
```python
# For high-resolution satellite images (>1000x1000)
patch_size = 25  # Larger patches for high-res
omega = 0.90     # Slightly less aggressive

# For low-resolution images (<500x500)  
patch_size = 7   # Smaller patches
omega = 0.95     # More aggressive

# For very hazy images
omega = 0.98     # Very aggressive dehazing
t0 = 0.05        # Lower minimum transmission

# For lightly hazy images
omega = 0.85     # Less aggressive
t0 = 0.2         # Higher minimum transmission
```

### CLAHE Parameters
```python
# For satellite images with fine details
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))

# For images with large uniform areas
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

# For very low contrast images
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
```

## 📊 Validation and Quality Check

### Always Validate Your Results
```python
def validate_dehazed_result(original, dehazed):
    """Validate that dehazing actually improved the image"""
    
    # Convert both to grayscale for analysis
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    else:
        orig_gray = original
        
    if len(dehazed.shape) == 3:
        dehazed_gray = cv2.cvtColor(dehazed, cv2.COLOR_RGB2GRAY)
    else:
        dehazed_gray = dehazed
    
    # Check contrast improvement
    orig_contrast = np.std(orig_gray)
    dehazed_contrast = np.std(dehazed_gray)
    contrast_ratio = dehazed_contrast / orig_contrast
    
    # Check edge enhancement
    orig_edges = cv2.Canny(orig_gray, 50, 150)
    dehazed_edges = cv2.Canny(dehazed_gray, 50, 150)
    
    orig_edge_density = np.sum(orig_edges > 0) / orig_edges.size
    dehazed_edge_density = np.sum(dehazed_edges > 0) / dehazed_edges.size
    
    print(f"Contrast improvement: {contrast_ratio:.2f}x")
    print(f"Edge enhancement: {dehazed_edge_density/orig_edge_density:.2f}x")
    
    # Quality indicators
    if contrast_ratio > 1.2 and dehazed_edge_density > orig_edge_density:
        print("✅ Good quality dehazing")
        return True
    elif contrast_ratio < 1.1:
        print("⚠️ Minimal improvement - try different parameters")
        return False
    else:
        print("✅ Moderate improvement")
        return True
```

## 🚨 Common Mistakes to Avoid

### 1. **Wrong Image Loading**
```python
# ❌ Wrong - may load as BGR
image = cv2.imread('satellite.jpg')

# ✅ Correct - ensure RGB
image = cv2.imread('satellite.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

### 2. **Incorrect Data Type Handling**
```python
# ❌ Wrong - may overflow or underflow
result = algorithm_output * 255

# ✅ Correct - proper clipping and conversion
result = np.clip(algorithm_output, 0, 1)
result = (result * 255).astype(np.uint8)
```

### 3. **Not Checking Output Range**
```python
# ❌ Wrong - assuming values are in correct range
cv2.imwrite('output.jpg', dehazed_result)

# ✅ Correct - validate before saving
print(f"Output range: [{dehazed_result.min()}, {dehazed_result.max()}]")
print(f"Output dtype: {dehazed_result.dtype}")
if dehazed_result.dtype != np.uint8:
    dehazed_result = (np.clip(dehazed_result, 0, 1) * 255).astype(np.uint8)
cv2.imwrite('output.jpg', dehazed_result)
```

## 🛰️ Satellite-Specific Considerations

### 1. **Band Selection**
- Use visible bands (Red, Green, Blue) for dehazing
- Consider Near-Infrared (NIR) for haze detection
- Avoid thermal bands for standard dehazing

### 2. **Atmospheric Correction First**
```python
# Consider atmospheric correction before dehazing
# This removes systematic atmospheric effects
def atmospheric_correction_first(image):
    # Simple dark object subtraction
    dark_pixel_value = np.percentile(image, 1, axis=(0,1))
    corrected = image - dark_pixel_value
    corrected = np.clip(corrected, 0, 255)
    return corrected.astype(np.uint8)
```

### 3. **Multi-temporal Analysis**
- Compare with clear-day images from same location
- Use temporal median filtering for persistent haze

## 🔍 Debugging Your Specific Case

### Step-by-Step Debugging Process
1. **Load and inspect original image**:
   ```python
   img = cv2.imread('your_satellite_image.jpg')
   print(f"Shape: {img.shape}, dtype: {img.dtype}")
   print(f"Range: [{img.min()}, {img.max()}]")
   ```

2. **Apply dehazing and check intermediate results**:
   ```python
   dehazed = your_dehazing_function(img)
   print(f"Dehazed - Shape: {dehazed.shape}, dtype: {dehazed.dtype}")
   print(f"Dehazed range: [{dehazed.min()}, {dehazed.max()}]")
   ```

3. **Save intermediate steps**:
   ```python
   cv2.imwrite('step1_loaded.jpg', img)
   cv2.imwrite('step2_dehazed_raw.jpg', dehazed)
   cv2.imwrite('step3_dehazed_normalized.jpg', normalize_for_display(dehazed))
   ```

4. **Visual comparison**:
   ```python
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(15, 5))
   plt.subplot(1, 3, 1); plt.imshow(img); plt.title('Original')
   plt.subplot(1, 3, 2); plt.imshow(dehazed); plt.title('Dehazed Raw')
   plt.subplot(1, 3, 3); plt.imshow(normalize_for_display(dehazed)); plt.title('Dehazed Normalized')
   plt.show()
   ```

## 📦 Ready-to-Use Solution

The files in this directory provide:

1. **`satellite_dehazing_debug.py`** - Comprehensive diagnostic tool
2. **`practical_dehazing_solution.py`** - Multiple working dehazing algorithms
3. **`dehaze_results_*/`** - Example outputs showing proper visibility

### To Use:
1. Place your satellite image in `/home/user/webapp/`
2. Name it `satellite_image.jpg` or `satellite_image.png`
3. Run: `python practical_dehazing_solution.py`
4. Check the `dehaze_results_*` directory for outputs

## 🎯 Summary

The most common reason dehazed satellite images are not visible is **improper data type and value range handling**. Always ensure:

1. ✅ Convert float results to uint8: `(np.clip(result, 0, 1) * 255).astype(np.uint8)`
2. ✅ Check value ranges before and after processing
3. ✅ Apply additional contrast enhancement if needed
4. ✅ Validate results with metrics, not just visual inspection
5. ✅ Use multiple dehazing methods and compare results

If you're still having issues, use the diagnostic tools provided to identify the specific problem with your images and processing pipeline.