import cv2
import numpy as np
from PIL import Image

class YOLOHeatmapGenerator:
    def __init__(self):
        pass

    def generate_heatmap(self, image_pil, yolo_results, alpha=0.5):
        """
        Generates a full-image heatmap based on YOLO detection boxes.
        Background = Blue (Cold)
        Issues = Red (Hot)
        """
        # Convert PIL to Numpy array (RGB)
        img = np.array(image_pil)
        
        h, w = img.shape[:2]
        
        # 1. Create a blank mask (zeros = cold/background)
        mask = np.zeros((h, w), dtype=np.float32)
        
        # 2. Draw "Hot" zones based on YOLO boxes
        has_issues = False
        if yolo_results and len(yolo_results) > 0:
            for box in yolo_results[0].boxes:
                # box.xyxy is [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw filled white rectangle on the mask (Value 1.0)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, -1)
                has_issues = True

        # 3. Create Gradients (Blur the sharp rectangles)
        if has_issues:
            # Kernel size roughly 10% of image width for smooth gradients
            k_size = int(w * 0.1) | 1 
            mask = cv2.GaussianBlur(mask, (k_size, k_size), 0)
            
            # Normalize peak to 1.0 (Red)
            if mask.max() > 0:
                mask /= mask.max()

        # 4. Apply ColorMap (JET: 0=Blue, 1=Red)
        heatmap_uint8 = (mask * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # OpenCV uses BGR, convert to RGB
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # 5. Blend Original Image with Heatmap
        # We overlay the heatmap color onto the original grayscale/RGB image
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
        
        return Image.fromarray(overlay)