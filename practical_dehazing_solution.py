#!/usr/bin/env python3
"""
Practical Satellite Image Dehazing Solution

This script provides working solutions for common dehazing visibility issues
and includes multiple dehazing algorithms with proper output handling.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage import exposure, filters
import warnings
warnings.filterwarnings('ignore')

class PracticalDehazingSolution:
    
    def __init__(self):
        self.debug = True
        
    def load_and_validate_image(self, image_path):
        """Load image and validate it's suitable for dehazing"""
        try:
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                return None
                
            # Try loading with different methods
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = np.array(Image.open(image_path))
            except:
                image = np.array(Image.open(image_path))
            
            if self.debug:
                print(f"✅ Image loaded: {image.shape}, dtype: {image.dtype}")
                print(f"   Value range: [{image.min()}, {image.max()}]")
                
            return image
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    
    def normalize_image_for_processing(self, image):
        """Normalize image to [0,1] float range for processing"""
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            # Already float, ensure [0,1] range
            return np.clip(image.astype(np.float32), 0, 1)
    
    def denormalize_for_display(self, image):
        """Convert processed float image back to uint8 for display/saving"""
        # Ensure values are in [0,1] range
        image_clipped = np.clip(image, 0, 1)
        
        # Convert to uint8
        return (image_clipped * 255).astype(np.uint8)
    
    def dark_channel_prior_dehazing(self, image, patch_size=15, omega=0.95, t0=0.1):
        """
        Dark Channel Prior dehazing algorithm
        
        Args:
            image: Input hazy image (float, 0-1 range)
            patch_size: Size of patches for dark channel computation
            omega: Parameter to keep some haze for natural look
            t0: Minimum transmission value
        """
        if self.debug:
            print("🔧 Applying Dark Channel Prior dehazing...")
            
        # Ensure 3 channels
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)
        
        # Calculate dark channel
        dark_channel = self.get_dark_channel(image, patch_size)
        
        # Estimate atmospheric light
        atmospheric_light = self.estimate_atmospheric_light(image, dark_channel)
        
        # Estimate transmission map
        transmission = 1 - omega * self.get_dark_channel(image / atmospheric_light, patch_size)
        transmission = np.maximum(transmission, t0)
        
        # Recover scene radiance
        dehazed = np.zeros_like(image)
        for i in range(3):
            dehazed[:,:,i] = (image[:,:,i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
        
        return np.clip(dehazed, 0, 1)
    
    def get_dark_channel(self, image, patch_size):
        """Calculate dark channel of the image"""
        if len(image.shape) == 3:
            dark = np.min(image, axis=2)
        else:
            dark = image
            
        kernel = np.ones((patch_size, patch_size))
        dark_channel = cv2.erode(dark, kernel)
        return dark_channel
    
    def estimate_atmospheric_light(self, image, dark_channel):
        """Estimate atmospheric light from brightest pixels in dark channel"""
        flat_dark = dark_channel.flatten()
        flat_image = image.reshape(-1, image.shape[2] if len(image.shape) == 3 else 1)
        
        # Get top 0.1% brightest pixels in dark channel
        num_pixels = int(0.001 * len(flat_dark))
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        
        # Get atmospheric light as max of these pixels
        if len(image.shape) == 3:
            atmospheric_light = np.max(flat_image[indices], axis=0)
        else:
            atmospheric_light = np.max(flat_image[indices])
            
        return atmospheric_light
    
    def histogram_equalization_dehazing(self, image):
        """Apply histogram equalization for dehazing effect"""
        if self.debug:
            print("🔧 Applying histogram equalization dehazing...")
            
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            from skimage.color import rgb2lab, lab2rgb
            
            lab = rgb2lab(image)
            # Apply histogram equalization to L channel
            lab[:,:,0] = exposure.equalize_hist(lab[:,:,0])
            result = lab2rgb(lab)
        else:
            result = exposure.equalize_hist(image)
            
        return np.clip(result, 0, 1)
    
    def contrast_enhancement_dehazing(self, image, gamma=0.6, alpha=1.2):
        """Apply contrast enhancement techniques"""
        if self.debug:
            print("🔧 Applying contrast enhancement dehazing...")
            
        # Gamma correction
        gamma_corrected = np.power(image, gamma)
        
        # Contrast stretching
        contrast_enhanced = exposure.rescale_intensity(gamma_corrected)
        
        # Apply additional contrast
        result = np.clip(alpha * contrast_enhanced, 0, 1)
        
        return result
    
    def adaptive_histogram_eq_dehazing(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if self.debug:
            print("🔧 Applying CLAHE dehazing...")
            
        # Convert to uint8 for CLAHE
        image_uint8 = (image * 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            # Apply CLAHE to each channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            result = np.zeros_like(image_uint8)
            for i in range(3):
                result[:,:,i] = clahe.apply(image_uint8[:,:,i])
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            result = clahe.apply(image_uint8)
            
        # Convert back to float
        return result.astype(np.float32) / 255.0
    
    def atmospheric_scattering_removal(self, image):
        """Simple atmospheric scattering removal"""
        if self.debug:
            print("🔧 Applying atmospheric scattering removal...")
            
        # Estimate atmospheric light as brightest pixels
        if len(image.shape) == 3:
            atmospheric_light = np.max(image.reshape(-1, 3), axis=0)
        else:
            atmospheric_light = np.max(image)
            
        # Simple transmission estimation
        transmission = 1 - 0.95 * np.min(image / atmospheric_light, axis=2 if len(image.shape) == 3 else None)
        transmission = np.maximum(transmission, 0.1)
        
        # Recover scene
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                result[:,:,i] = (image[:,:,i] - atmospheric_light[i]) / np.expand_dims(transmission, axis=2)[:,:,0] + atmospheric_light[i]
        else:
            result = (image - atmospheric_light) / transmission + atmospheric_light
            
        return np.clip(result, 0, 1)
    
    def process_image_with_multiple_methods(self, image_path):
        """Process image with multiple dehazing methods and ensure visibility"""
        
        print(f"🛰️ Processing satellite image: {os.path.basename(image_path)}")
        print("=" * 60)
        
        # Load and validate image
        original = self.load_and_validate_image(image_path)
        if original is None:
            return None
            
        # Normalize for processing
        image_normalized = self.normalize_image_for_processing(original)
        
        # Apply multiple dehazing methods
        methods = {
            'dark_channel_prior': self.dark_channel_prior_dehazing,
            'histogram_equalization': self.histogram_equalization_dehazing,
            'contrast_enhancement': self.contrast_enhancement_dehazing,
            'adaptive_histogram_eq': self.adaptive_histogram_eq_dehazing,
            'atmospheric_scattering': self.atmospheric_scattering_removal
        }
        
        results = {}
        
        for method_name, method_func in methods.items():
            try:
                print(f"\n--- Processing with {method_name.replace('_', ' ').title()} ---")
                
                # Apply method
                result_normalized = method_func(image_normalized.copy())
                
                # Convert to displayable format
                result_uint8 = self.denormalize_for_display(result_normalized)
                
                # Validate result
                if self.validate_result(result_uint8):
                    results[method_name] = {
                        'normalized': result_normalized,
                        'uint8': result_uint8
                    }
                    print(f"✅ {method_name} completed successfully")
                else:
                    print(f"⚠️ {method_name} produced low-quality result")
                    
            except Exception as e:
                print(f"❌ Error in {method_name}: {e}")
                continue
        
        # Save results
        if results:
            self.save_results(original, results, image_path)
            self.create_comparison_visualization(original, results, image_path)
        
        return results
    
    def validate_result(self, result):
        """Validate that the dehazing result is visible and reasonable"""
        
        # Check for basic issues
        if np.all(result == 0) or np.all(result == 255):
            return False
        
        # Check contrast
        if np.std(result) < 10:
            return False
            
        # Check dynamic range
        if (np.max(result) - np.min(result)) < 50:
            return False
            
        return True
    
    def save_results(self, original, results, original_path):
        """Save all results with proper naming"""
        
        # Create output directory
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_dir = f"/home/user/webapp/dehaze_results_{base_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original
        Image.fromarray(original).save(os.path.join(output_dir, "original.png"))
        
        print(f"\n💾 SAVING RESULTS TO: {output_dir}")
        print("-" * 50)
        
        # Save each result
        for method_name, result_data in results.items():
            output_path = os.path.join(output_dir, f"{method_name}_dehazed.png")
            Image.fromarray(result_data['uint8']).save(output_path)
            print(f"✅ Saved: {method_name}_dehazed.png")
        
        # Create a side-by-side comparison
        self.create_side_by_side_comparison(original, results, output_dir)
        
        print(f"\n📁 All results saved in: {output_dir}")
    
    def create_side_by_side_comparison(self, original, results, output_dir):
        """Create side-by-side comparison of all methods"""
        
        n_methods = len(results) + 1  # +1 for original
        cols = 3
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Plot original
        axes_flat[0].imshow(original)
        axes_flat[0].set_title('Original (Hazy)', fontsize=12, fontweight='bold')
        axes_flat[0].axis('off')
        
        # Plot results
        for idx, (method_name, result_data) in enumerate(results.items(), 1):
            if idx < len(axes_flat):
                axes_flat[idx].imshow(result_data['uint8'])
                title = method_name.replace('_', ' ').title()
                axes_flat[idx].set_title(f'{title} (Dehazed)', fontsize=12, fontweight='bold')
                axes_flat[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(len(results) + 1, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, 'comparison_all_methods.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved comparison: comparison_all_methods.png")
    
    def create_comparison_visualization(self, original, results, original_path):
        """Create detailed comparison visualization"""
        pass  # Implementation would go here for detailed analysis
    
    def create_demo_hazy_satellite_image(self):
        """Create a realistic demo hazy satellite image"""
        print("📸 Creating demo hazy satellite image...")
        
        # Create a more realistic satellite-like image
        size = (512, 512)
        
        # Create base terrain features
        np.random.seed(42)
        
        # Create landforms using noise
        x = np.linspace(0, 10, size[1])
        y = np.linspace(0, 10, size[0])
        X, Y = np.meshgrid(x, y)
        
        # Base terrain
        terrain = 0.3 + 0.2 * np.sin(X * 0.5) * np.cos(Y * 0.7)
        terrain += 0.1 * np.random.random(size)
        
        # Add field patterns
        fields = 0.2 * (np.sin(X * 2) > 0.5) * (np.cos(Y * 1.5) > 0.2)
        
        # Add urban areas
        urban = 0.4 * ((X - 5)**2 + (Y - 5)**2 < 4) * np.random.random(size)
        
        # Combine features
        clear_scene = np.clip(terrain + fields + urban, 0, 1)
        
        # Create 3-channel image
        clear_rgb = np.stack([
            clear_scene * 0.8,  # Red
            clear_scene * 0.9,  # Green  
            clear_scene * 0.7   # Blue
        ], axis=2)
        
        # Add haze effect (atmospheric scattering)
        atmospheric_light = np.array([0.85, 0.9, 0.95])  # Bluish-white
        
        # Create non-uniform haze (more realistic)
        haze_density = 0.6 + 0.2 * np.sin(X * 0.3) * np.cos(Y * 0.4)
        haze_density = np.clip(haze_density, 0.4, 0.8)
        
        # Apply haze model: I = J * t + A * (1 - t)
        # where t is transmission, J is scene radiance, A is atmospheric light
        transmission = 1 - haze_density
        transmission = np.expand_dims(transmission, axis=2)
        
        hazy_image = clear_rgb * transmission + atmospheric_light * (1 - transmission)
        hazy_image = np.clip(hazy_image, 0, 1)
        
        # Convert to uint8 and save
        hazy_uint8 = (hazy_image * 255).astype(np.uint8)
        demo_path = '/home/user/webapp/demo_hazy_satellite.png'
        Image.fromarray(hazy_uint8).save(demo_path)
        
        print(f"✅ Demo image created: {demo_path}")
        return demo_path

def main():
    """Main function to demonstrate the dehazing solution"""
    
    print("🛰️ PRACTICAL SATELLITE DEHAZING SOLUTION")
    print("=" * 60)
    
    solver = PracticalDehazingSolution()
    
    # Look for existing images first
    test_images = [
        '/home/user/webapp/satellite_image.png',
        '/home/user/webapp/satellite_image.jpg',
        '/home/user/webapp/sample_hazy_satellite.png',
        '/home/user/webapp/hazy_image.png'
    ]
    
    image_found = False
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"📸 Found existing image: {img_path}")
            results = solver.process_image_with_multiple_methods(img_path)
            image_found = True
            break
    
    if not image_found:
        print("📸 No existing images found, creating demo...")
        demo_path = solver.create_demo_hazy_satellite_image()
        results = solver.process_image_with_multiple_methods(demo_path)
    
    # Print final summary
    print("\n🎯 SUMMARY & NEXT STEPS")
    print("=" * 50)
    
    if results:
        print(f"✅ Successfully processed image with {len(results)} methods")
        print("📁 Check the 'dehaze_results_*' directory for:")
        print("   • Individual dehazed images")
        print("   • Side-by-side comparison")
        print("   • Original image for reference")
        
        print("\n🔍 TROUBLESHOOTING TIPS:")
        print("1. If results still look poor, your image may need preprocessing")
        print("2. Try adjusting the method parameters for your specific image type")
        print("3. Ensure your display software can handle the image format correctly")
        print("4. Check that file permissions allow reading the output images")
        print("5. For satellite imagery, consider atmospheric correction first")
        
    else:
        print("❌ No successful dehazing results produced")
        print("This could indicate:")
        print("   • Input image has issues")
        print("   • Image is not actually hazy") 
        print("   • Algorithm parameters need adjustment")

if __name__ == "__main__":
    main()