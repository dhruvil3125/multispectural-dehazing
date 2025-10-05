#!/usr/bin/env python3
"""
Comprehensive Satellite Image Dehazing Troubleshooting Guide

This script provides detailed troubleshooting for common issues when
dehazed satellite images are not visible or appear incorrect.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

class DehazingTroubleshooter:
    def __init__(self):
        self.common_issues = {
            "data_type_mismatch": {
                "description": "Output image has wrong data type for display",
                "symptoms": ["Black/white image", "No visible content", "Extreme values"],
                "solutions": [
                    "Convert float images to uint8: (image * 255).astype(np.uint8)",
                    "Normalize values to [0,1] range before conversion",
                    "Check if values are in expected range for data type"
                ]
            },
            "value_range_error": {
                "description": "Image values outside displayable range",
                "symptoms": ["All black image", "All white image", "No contrast"],
                "solutions": [
                    "Clip values to valid range: np.clip(image, 0, 255) for uint8",
                    "Normalize: (image - image.min()) / (image.max() - image.min())",
                    "Use cv2.normalize() for automatic range adjustment"
                ]
            },
            "insufficient_contrast": {
                "description": "Dehazing didn't improve visibility enough",
                "symptoms": ["Subtle differences", "Still hazy appearance", "Minimal improvement"],
                "solutions": [
                    "Apply histogram equalization after dehazing",
                    "Use adaptive contrast enhancement",
                    "Try different dehazing parameters",
                    "Combine multiple enhancement techniques"
                ]
            },
            "algorithm_parameters": {
                "description": "Dehazing algorithm parameters not optimal",
                "symptoms": ["Over-correction", "Under-correction", "Artifacts"],
                "solutions": [
                    "Adjust transmission map parameters",
                    "Modify atmospheric light estimation",
                    "Fine-tune guided filter parameters",
                    "Test with different patch sizes"
                ]
            },
            "color_space_issues": {
                "description": "Wrong color space for processing or display",
                "symptoms": ["Color distortion", "Channel misalignment", "Unusual colors"],
                "solutions": [
                    "Ensure RGB channel order (not BGR)",
                    "Check color space conversion (RGB/LAB/HSV)",
                    "Verify band ordering for satellite imagery"
                ]
            }
        }
    
    def diagnose_invisible_result(self, original_path, dehazed_path=None, dehazed_array=None):
        """Comprehensive diagnosis of invisible dehazing results"""
        
        print("🔍 DEHAZING INVISIBILITY DIAGNOSTIC")
        print("=" * 60)
        
        # Load original image
        if not os.path.exists(original_path):
            print(f"❌ Original image not found: {original_path}")
            return
        
        original = np.array(Image.open(original_path))
        print(f"✅ Original image loaded: {original.shape}, {original.dtype}")
        
        # Load or use provided dehazed image
        dehazed = None
        if dehazed_path and os.path.exists(dehazed_path):
            dehazed = np.array(Image.open(dehazed_path))
            print(f"✅ Dehazed image loaded: {dehazed.shape}, {dehazed.dtype}")
        elif dehazed_array is not None:
            dehazed = dehazed_array
            print(f"✅ Dehazed array provided: {dehazed.shape}, {dehazed.dtype}")
        else:
            print("⚠️  No dehazed image provided - will analyze potential issues only")
        
        # Run diagnostics
        self._check_data_types(original, dehazed)
        self._check_value_ranges(original, dehazed)
        self._check_visibility_metrics(original, dehazed)
        self._provide_specific_solutions(original, dehazed)
    
    def _check_data_types(self, original, dehazed):
        """Check for data type related issues"""
        print("\n🔬 DATA TYPE ANALYSIS")
        print("-" * 30)
        
        print(f"Original: {original.dtype}, range [{original.min()}, {original.max()}]")
        
        if dehazed is not None:
            print(f"Dehazed: {dehazed.dtype}, range [{dehazed.min()}, {dehazed.max()}]")
            
            # Check for common data type issues
            if original.dtype != dehazed.dtype:
                print("⚠️  WARNING: Data type mismatch between original and dehazed")
            
            if dehazed.dtype == np.float32 or dehazed.dtype == np.float64:
                if dehazed.max() > 1.0:
                    print("⚠️  WARNING: Float image with values > 1.0 - may not display correctly")
                elif dehazed.max() <= 1.0 and dehazed.min() >= 0:
                    print("✅ Float image properly normalized [0,1]")
            
            if dehazed.dtype == np.uint8:
                if np.all(dehazed == 0):
                    print("❌ ERROR: All pixels are 0 - image appears black")
                elif np.all(dehazed == 255):
                    print("❌ ERROR: All pixels are 255 - image appears white")
    
    def _check_value_ranges(self, original, dehazed):
        """Check for value range issues"""
        print("\n📊 VALUE RANGE ANALYSIS")
        print("-" * 30)
        
        if dehazed is not None:
            # Calculate statistics
            orig_stats = self._get_image_stats(original)
            dehazed_stats = self._get_image_stats(dehazed)
            
            print("Original statistics:")
            self._print_stats(orig_stats)
            
            print("\nDehazed statistics:")
            self._print_stats(dehazed_stats)
            
            # Check for problematic ranges
            if dehazed_stats['std'] < 5 and dehazed.dtype == np.uint8:
                print("⚠️  WARNING: Very low standard deviation - image may lack contrast")
            
            if abs(dehazed_stats['mean'] - orig_stats['mean']) < 5:
                print("⚠️  WARNING: Minimal change in mean brightness - dehazing may be ineffective")
    
    def _check_visibility_metrics(self, original, dehazed):
        """Check visibility improvement metrics"""
        print("\n👁️  VISIBILITY METRICS")
        print("-" * 30)
        
        if dehazed is None:
            print("Cannot calculate - no dehazed image provided")
            return
        
        # Convert to grayscale for analysis
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        else:
            orig_gray = original
            
        if len(dehazed.shape) == 3:
            dehazed_gray = cv2.cvtColor(dehazed.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            dehazed_gray = dehazed.astype(np.uint8)
        
        # Calculate contrast improvement
        orig_contrast = np.std(orig_gray)
        dehazed_contrast = np.std(dehazed_gray)
        contrast_improvement = dehazed_contrast / orig_contrast
        
        print(f"Original contrast (std): {orig_contrast:.2f}")
        print(f"Dehazed contrast (std): {dehazed_contrast:.2f}")
        print(f"Contrast improvement ratio: {contrast_improvement:.2f}")
        
        if contrast_improvement < 1.1:
            print("⚠️  WARNING: Minimal contrast improvement - results may not be visibly different")
        elif contrast_improvement > 3.0:
            print("⚠️  WARNING: Excessive contrast change - may indicate over-processing")
        else:
            print("✅ Reasonable contrast improvement")
        
        # Calculate edge enhancement
        orig_edges = cv2.Canny(orig_gray, 50, 150)
        dehazed_edges = cv2.Canny(dehazed_gray, 50, 150)
        
        orig_edge_density = np.sum(orig_edges > 0) / orig_edges.size
        dehazed_edge_density = np.sum(dehazed_edges > 0) / dehazed_edges.size
        
        print(f"Original edge density: {orig_edge_density:.4f}")
        print(f"Dehazed edge density: {dehazed_edge_density:.4f}")
        
        if dehazed_edge_density > orig_edge_density * 1.2:
            print("✅ Good edge enhancement")
        else:
            print("⚠️  Limited edge enhancement - details may not be significantly clearer")
    
    def _get_image_stats(self, image):
        """Calculate image statistics"""
        if len(image.shape) == 3:
            # Convert to grayscale for overall statistics
            gray = np.mean(image, axis=2)
        else:
            gray = image
            
        return {
            'mean': np.mean(gray),
            'std': np.std(gray),
            'min': np.min(gray),
            'max': np.max(gray),
            'median': np.median(gray)
        }
    
    def _print_stats(self, stats):
        """Print image statistics"""
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Std: {stats['std']:.2f}")
        print(f"  Min: {stats['min']:.2f}")
        print(f"  Max: {stats['max']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
    
    def _provide_specific_solutions(self, original, dehazed):
        """Provide specific solutions based on analysis"""
        print("\n💡 SPECIFIC SOLUTIONS")
        print("-" * 30)
        
        if dehazed is None:
            self._print_general_solutions()
            return
        
        solutions = []
        
        # Check for specific issues and provide targeted solutions
        if dehazed.dtype in [np.float32, np.float64] and dehazed.max() > 1.0:
            solutions.append("NORMALIZE VALUES: dehazed = np.clip(dehazed, 0, 1)")
        
        if dehazed.dtype in [np.float32, np.float64]:
            solutions.append("CONVERT TO UINT8: dehazed_uint8 = (dehazed * 255).astype(np.uint8)")
        
        dehazed_stats = self._get_image_stats(dehazed)
        if dehazed_stats['std'] < 10:
            solutions.append("ENHANCE CONTRAST: dehazed = cv2.equalizeHist(dehazed)")
        
        if abs(dehazed_stats['mean'] - 128) > 100:
            solutions.append("ADJUST BRIGHTNESS: dehazed = cv2.normalize(dehazed, None, 0, 255, cv2.NORM_MINMAX)")
        
        # Print solutions
        if solutions:
            print("Recommended fixes:")
            for i, solution in enumerate(solutions, 1):
                print(f"  {i}. {solution}")
        else:
            print("No specific issues detected. Try general enhancement techniques.")
    
    def _print_general_solutions(self):
        """Print general solutions for common issues"""
        print("General troubleshooting steps:")
        print("1. Verify image is loaded correctly")
        print("2. Check data type consistency (uint8 vs float)")
        print("3. Ensure proper value range (0-255 for uint8, 0-1 for float)")
        print("4. Apply contrast enhancement after dehazing")
        print("5. Try different dehazing algorithm parameters")
        print("6. Use visualization tools to compare results")

def create_dehazing_examples():
    """Create examples of common dehazing issues and solutions"""
    
    print("🛠️ CREATING DEHAZING EXAMPLES")
    print("=" * 50)
    
    # Create sample hazy image
    np.random.seed(42)
    size = (256, 256, 3)
    
    # Base image with some structure
    x = np.linspace(0, 4*np.pi, size[1])
    y = np.linspace(0, 4*np.pi, size[0])
    X, Y = np.meshgrid(x, y)
    
    base = 0.3 + 0.4 * (np.sin(X) + np.cos(Y)) + 0.3 * np.random.rand(*size)
    
    # Add haze
    haze = 0.6 * np.ones_like(base)
    hazy = 0.4 * base + 0.6 * haze
    hazy_uint8 = (np.clip(hazy, 0, 1) * 255).astype(np.uint8)
    
    # Save original hazy image
    Image.fromarray(hazy_uint8).save('/home/user/webapp/example_hazy.png')
    print("✅ Created example hazy image")
    
    # Create problematic dehazing results
    examples = {}
    
    # Example 1: Wrong data type (float values > 1)
    dehazed_wrong_range = base * 2.5  # Values > 1
    examples['wrong_range'] = dehazed_wrong_range
    
    # Example 2: All zeros (black image)
    dehazed_black = np.zeros_like(base)
    examples['black'] = dehazed_black
    
    # Example 3: All ones (white image)
    dehazed_white = np.ones_like(base)
    examples['white'] = dehazed_white
    
    # Example 4: Low contrast
    dehazed_low_contrast = 0.5 + 0.1 * (base - 0.5)
    examples['low_contrast'] = dehazed_low_contrast
    
    # Example 5: Proper dehazing
    dehazed_proper = np.clip(1.5 * base - 0.3, 0, 1)
    examples['proper'] = dehazed_proper
    
    # Save examples and analyze
    for name, dehazed in examples.items():
        # Save as different formats to show the issues
        if name == 'wrong_range':
            # Save with wrong range (will clip)
            Image.fromarray(np.clip(dehazed * 255, 0, 255).astype(np.uint8)).save(f'/home/user/webapp/example_{name}.png')
        else:
            # Save properly
            Image.fromarray((np.clip(dehazed, 0, 1) * 255).astype(np.uint8)).save(f'/home/user/webapp/example_{name}.png')
        
        print(f"✅ Created example: {name}")
    
    return '/home/user/webapp/example_hazy.png', examples

def main():
    """Main function"""
    print("🔧 SATELLITE IMAGE DEHAZING TROUBLESHOOTER")
    print("=" * 60)
    
    # Create examples
    hazy_path, examples = create_dehazing_examples()
    
    # Initialize troubleshooter
    troubleshooter = DehazingTroubleshooter()
    
    print(f"\n🔍 Analyzing example issues:")
    print("-" * 40)
    
    # Analyze each problematic example
    for name, dehazed_array in examples.items():
        print(f"\n--- ANALYZING: {name.upper()} ---")
        troubleshooter.diagnose_invisible_result(hazy_path, dehazed_array=dehazed_array)
        print("\n" + "="*50)
    
    print("\n📚 COMMON ISSUE REFERENCE")
    print("=" * 50)
    
    for issue_name, issue_info in troubleshooter.common_issues.items():
        print(f"\n🔸 {issue_name.replace('_', ' ').upper()}")
        print(f"Description: {issue_info['description']}")
        print(f"Symptoms: {', '.join(issue_info['symptoms'])}")
        print("Solutions:")
        for solution in issue_info['solutions']:
            print(f"  • {solution}")

if __name__ == "__main__":
    main()