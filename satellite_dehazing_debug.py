#!/usr/bin/env python3
"""
Satellite Image Dehazing Diagnostic Tool

This script helps diagnose common issues with satellite image dehazing
where the processed image is not visible or appears incorrect.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
from skimage import exposure, filters, restoration
import warnings
warnings.filterwarnings('ignore')

class SatelliteDehazeDebugger:
    def __init__(self):
        self.original_image = None
        self.dehazed_image = None
        self.image_path = None
        
    def load_image(self, image_path):
        """Load and validate satellite image"""
        print(f"Loading image from: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found at {image_path}")
            return False
            
        try:
            # Try loading with PIL first
            self.original_image = np.array(Image.open(image_path))
            self.image_path = image_path
            
            print(f"✅ Image loaded successfully")
            print(f"   Shape: {self.original_image.shape}")
            print(f"   Data type: {self.original_image.dtype}")
            print(f"   Value range: [{self.original_image.min()}, {self.original_image.max()}]")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return False
    
    def analyze_image_properties(self):
        """Analyze image properties that affect dehazing visibility"""
        if self.original_image is None:
            print("❌ No image loaded")
            return
            
        print("\n📊 IMAGE ANALYSIS")
        print("=" * 50)
        
        # Check if image is grayscale or color
        if len(self.original_image.shape) == 2:
            print("🔍 Image type: Grayscale")
            channels = 1
        elif len(self.original_image.shape) == 3:
            channels = self.original_image.shape[2]
            print(f"🔍 Image type: Color ({channels} channels)")
        else:
            print("❌ Unexpected image dimensions")
            return
            
        # Analyze dynamic range
        print(f"📈 Dynamic range analysis:")
        if channels == 1:
            img_flat = self.original_image.flatten()
            print(f"   Min value: {img_flat.min()}")
            print(f"   Max value: {img_flat.max()}")
            print(f"   Mean: {img_flat.mean():.2f}")
            print(f"   Std dev: {img_flat.std():.2f}")
            
            # Check for low contrast (common issue)
            if img_flat.std() < 20 and self.original_image.dtype == np.uint8:
                print("⚠️  WARNING: Very low contrast detected - may affect dehazing visibility")
                
        else:
            for i in range(min(3, channels)):
                channel = self.original_image[:,:,i]
                print(f"   Channel {i}: range [{channel.min()}, {channel.max()}], mean: {channel.mean():.2f}")
        
        # Check data type issues
        if self.original_image.dtype == np.uint8:
            print("📝 Data type: 8-bit (0-255 range)")
        elif self.original_image.dtype == np.uint16:
            print("📝 Data type: 16-bit (0-65535 range)")
        elif self.original_image.dtype == np.float32 or self.original_image.dtype == np.float64:
            print(f"📝 Data type: Float ({self.original_image.dtype})")
            if self.original_image.max() <= 1.0:
                print("   Values appear to be normalized (0-1 range)")
            else:
                print("   Values are not normalized")
    
    def check_common_issues(self):
        """Check for common issues that prevent dehazed image visibility"""
        if self.original_image is None:
            return
            
        print("\n🔍 COMMON ISSUES CHECK")
        print("=" * 50)
        
        issues_found = []
        
        # Issue 1: Data type mismatch
        if self.original_image.dtype not in [np.uint8, np.float32, np.float64]:
            issues_found.append("Unusual data type - may cause display issues")
        
        # Issue 2: Value range issues
        if self.original_image.dtype == np.float32 or self.original_image.dtype == np.float64:
            if self.original_image.max() > 1.0:
                issues_found.append("Float values > 1.0 - may need normalization for display")
        
        # Issue 3: Very dark image
        mean_brightness = np.mean(self.original_image)
        if self.original_image.dtype == np.uint8 and mean_brightness < 50:
            issues_found.append("Very dark image - dehazing effects may not be visible")
        elif self.original_image.dtype in [np.float32, np.float64] and mean_brightness < 0.2:
            issues_found.append("Very dark image - dehazing effects may not be visible")
        
        # Issue 4: Low contrast
        if len(self.original_image.shape) == 3:
            gray = cv2.cvtColor(self.original_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = self.original_image.astype(np.uint8)
            
        contrast = np.std(gray)
        if contrast < 15:
            issues_found.append("Very low contrast - dehazing may have minimal visible effect")
        
        # Issue 5: Saturation
        if np.any(self.original_image == 255) and self.original_image.dtype == np.uint8:
            saturated_pixels = np.sum(self.original_image == 255)
            total_pixels = self.original_image.size
            if saturated_pixels / total_pixels > 0.1:
                issues_found.append("High number of saturated pixels (255 values)")
        
        # Report issues
        if issues_found:
            print("⚠️  Issues found:")
            for i, issue in enumerate(issues_found, 1):
                print(f"   {i}. {issue}")
        else:
            print("✅ No obvious issues detected")
    
    def apply_multiple_dehazing_methods(self):
        """Apply multiple dehazing methods and compare results"""
        if self.original_image is None:
            print("❌ No image loaded")
            return
            
        print("\n🛠️  APPLYING DEHAZING METHODS")
        print("=" * 50)
        
        # Convert to float for processing
        if self.original_image.dtype == np.uint8:
            img_float = self.original_image.astype(np.float32) / 255.0
        else:
            img_float = self.original_image.astype(np.float32)
            
        methods_results = {}
        
        try:
            # Method 1: Simple contrast enhancement
            print("🔧 Applying contrast enhancement...")
            enhanced = exposure.rescale_intensity(img_float)
            methods_results['contrast_enhanced'] = enhanced
            
            # Method 2: Histogram equalization
            print("🔧 Applying histogram equalization...")
            if len(img_float.shape) == 3:
                # Convert to LAB for better color preservation
                from skimage.color import rgb2lab, lab2rgb
                lab = rgb2lab(img_float)
                lab[:,:,0] = exposure.equalize_hist(lab[:,:,0])
                hist_eq = lab2rgb(lab)
            else:
                hist_eq = exposure.equalize_adaptive(img_float)
            methods_results['histogram_eq'] = hist_eq
            
            # Method 3: Gamma correction
            print("🔧 Applying gamma correction...")
            gamma_corrected = exposure.adjust_gamma(img_float, gamma=0.6)
            methods_results['gamma_corrected'] = gamma_corrected
            
            # Method 4: Unsharp masking (edge enhancement)
            print("🔧 Applying unsharp masking...")
            if len(img_float.shape) == 3:
                gray_float = np.mean(img_float, axis=2)
            else:
                gray_float = img_float
                
            gaussian = filters.gaussian(gray_float, sigma=1.0)
            unsharp = gray_float + 0.5 * (gray_float - gaussian)
            
            if len(img_float.shape) == 3:
                unsharp_color = img_float.copy()
                for i in range(3):
                    unsharp_color[:,:,i] = img_float[:,:,i] + 0.5 * (img_float[:,:,i] - filters.gaussian(img_float[:,:,i], sigma=1.0))
                methods_results['unsharp'] = np.clip(unsharp_color, 0, 1)
            else:
                methods_results['unsharp'] = np.clip(unsharp, 0, 1)
            
            # Method 5: Dark channel prior (simple approximation)
            print("🔧 Applying dark channel enhancement...")
            if len(img_float.shape) == 3:
                dark_channel = np.min(img_float, axis=2)
                transmission = 1 - 0.95 * dark_channel
                transmission = np.maximum(transmission, 0.1)
                
                dehazed = np.zeros_like(img_float)
                for i in range(3):
                    dehazed[:,:,i] = (img_float[:,:,i] - dark_channel) / transmission + dark_channel
                
                methods_results['dark_channel'] = np.clip(dehazed, 0, 1)
            
            print(f"✅ Applied {len(methods_results)} dehazing methods successfully")
            
        except Exception as e:
            print(f"❌ Error in dehazing: {e}")
            return None
            
        return methods_results
    
    def save_results_and_visualize(self, methods_results):
        """Save results and create visualization"""
        if methods_results is None or not methods_results:
            return
            
        print("\n💾 SAVING RESULTS")
        print("=" * 50)
        
        # Create output directory
        output_dir = "/home/user/webapp/dehazing_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual results
        for method_name, result in methods_results.items():
            # Convert to uint8 for saving
            if result.dtype != np.uint8:
                result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
            else:
                result_uint8 = result
                
            output_path = os.path.join(output_dir, f"{method_name}_result.png")
            Image.fromarray(result_uint8).save(output_path)
            print(f"✅ Saved: {output_path}")
        
        # Create comparison visualization
        self.create_comparison_plot(methods_results, output_dir)
        
        print(f"\n📁 All results saved to: {output_dir}")
    
    def create_comparison_plot(self, methods_results, output_dir):
        """Create a comparison plot of all methods"""
        print("🎨 Creating comparison visualization...")
        
        try:
            n_methods = len(methods_results) + 1  # +1 for original
            cols = min(3, n_methods)
            rows = (n_methods + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1 or cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            # Plot original image
            ax_idx = 0
            if len(self.original_image.shape) == 3:
                axes[ax_idx].imshow(self.original_image)
            else:
                axes[ax_idx].imshow(self.original_image, cmap='gray')
            axes[ax_idx].set_title('Original Image')
            axes[ax_idx].axis('off')
            ax_idx += 1
            
            # Plot results
            for method_name, result in methods_results.items():
                if ax_idx < len(axes):
                    if len(result.shape) == 3:
                        axes[ax_idx].imshow(np.clip(result, 0, 1))
                    else:
                        axes[ax_idx].imshow(np.clip(result, 0, 1), cmap='gray')
                    axes[ax_idx].set_title(method_name.replace('_', ' ').title())
                    axes[ax_idx].axis('off')
                    ax_idx += 1
            
            # Hide unused subplots
            for i in range(ax_idx, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, 'dehazing_comparison.png')
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Comparison plot saved: {comparison_path}")
            
        except Exception as e:
            print(f"❌ Error creating comparison plot: {e}")
    
    def provide_recommendations(self):
        """Provide recommendations based on analysis"""
        print("\n💡 RECOMMENDATIONS")
        print("=" * 50)
        
        recommendations = [
            "1. Check image format and data type consistency",
            "2. Ensure proper value range (0-255 for uint8, 0-1 for float)",
            "3. Try multiple dehazing algorithms - different methods work better for different conditions",
            "4. Consider preprocessing: noise reduction, contrast enhancement",
            "5. Validate output image properties (range, data type, dimensions)",
            "6. For satellite images, consider atmospheric correction before dehazing",
            "7. Check if the 'haze' is actually clouds, fog, or atmospheric scattering",
            "8. Experiment with different parameters in your dehazing algorithm",
            "9. Use visualization tools to compare before/after results side-by-side",
            "10. Consider using specialized satellite image processing libraries (GDAL, rasterio)"
        ]
        
        for rec in recommendations:
            print(f"   {rec}")

def main():
    """Main function to run the diagnostic tool"""
    print("🛰️  SATELLITE IMAGE DEHAZING DIAGNOSTIC TOOL")
    print("=" * 60)
    
    debugger = SatelliteDehazeDebugger()
    
    # Check for test images or ask for image path
    test_image_paths = [
        "/home/user/webapp/satellite_image.jpg",
        "/home/user/webapp/satellite_image.png",
        "/home/user/webapp/satellite_image.tif",
        "/home/user/webapp/input.jpg",
        "/home/user/webapp/input.png",
        "/home/user/webapp/image.jpg",
        "/home/user/webapp/image.png"
    ]
    
    image_loaded = False
    for path in test_image_paths:
        if os.path.exists(path):
            print(f"Found image: {path}")
            if debugger.load_image(path):
                image_loaded = True
                break
    
    if not image_loaded:
        print("No test images found. You can:")
        print("1. Place your satellite image in /home/user/webapp/")
        print("2. Name it 'satellite_image.jpg' or 'satellite_image.png'")
        print("3. Run this script again")
        print("\nFor now, I'll create a sample hazy image for demonstration...")
        
        # Create a sample hazy satellite-like image
        create_sample_hazy_image(debugger)
    
    if debugger.original_image is not None:
        # Run full analysis
        debugger.analyze_image_properties()
        debugger.check_common_issues()
        
        # Apply dehazing methods
        results = debugger.apply_multiple_dehazing_methods()
        
        if results:
            debugger.save_results_and_visualize(results)
        
        debugger.provide_recommendations()

def create_sample_hazy_image(debugger):
    """Create a sample hazy satellite-like image for testing"""
    print("\n📸 Creating sample hazy satellite image...")
    
    # Create synthetic satellite-like image with haze
    size = (512, 512, 3)
    
    # Create base landscape-like pattern
    np.random.seed(42)
    base = np.random.rand(*size) * 0.6
    
    # Add some structure (like fields, roads, etc.)
    x = np.linspace(0, 10, size[1])
    y = np.linspace(0, 10, size[0])
    X, Y = np.meshgrid(x, y)
    
    # Add periodic patterns (like agricultural fields)
    pattern = 0.3 * np.sin(X) * np.sin(Y) + 0.2 * np.cos(2*X) * np.cos(2*Y)
    
    for i in range(3):
        base[:,:,i] += pattern * (0.8 + 0.4 * np.random.rand())
    
    # Add haze effect
    haze_strength = 0.6
    haze_color = np.array([0.8, 0.85, 0.9])  # Bluish-white haze
    
    hazy_image = base * (1 - haze_strength) + haze_strength * haze_color
    
    # Convert to uint8
    hazy_image = np.clip(hazy_image * 255, 0, 255).astype(np.uint8)
    
    # Save sample image
    sample_path = "/home/user/webapp/sample_hazy_satellite.png"
    Image.fromarray(hazy_image).save(sample_path)
    print(f"✅ Sample hazy image created: {sample_path}")
    
    # Load it into debugger
    debugger.load_image(sample_path)

if __name__ == "__main__":
    main()