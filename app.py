import os
import torch
import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
from torchvision import transforms
from transformers import (
    SegformerImageProcessor, 
    SegformerForSemanticSegmentation,
    ViTImageProcessor, 
    ViTForImageClassification
)
from google import genai
from google.genai import types
from ultralytics import YOLO
from collections import Counter

# Import the new Heatmap class
from utils.yolo_heatmap import YOLOHeatmapGenerator


""" OUTPUT will display Yolo analysis on Left with hear mapped of yolo analysis , Yolo box on right , 
Classification using Vit on left bottom, AI Generated output on right bottom"""


# CONFIG & CONSTANTS

YOLO_CLASSES = {
    0: "Caries", 1: "Crown", 2: "Filling", 3: "Implant",
    4: "Malaligned", 5: "Mandibular Canal", 6: "Missing Teeth",
    7: "Periapical Lesion", 8: "Retained Root", 9: "RCT",
    10: "Root Piece", 11: "Impacted Tooth", 12: "Maxillary Sinus",
    13: "Bone Loss", 14: "Fracture Teeth", 15: "Permanent Teeth",
    16: "Supra Eruption", 17: "TAD", 18: "Abutment",
    19: "Attrition", 20: "Bone Defect", 21: "Gingival Former",
    22: "Metal Band", 23: "Orthodontic Brackets",
    24: "Permanent Retainer", 25: "Post-Core",
    26: "Plating", 27: "Wire", 28: "Cyst",
    29: "Root Resorption", 30: "Primary Teeth"
}

ISSUE_CLASSES_YOLO = set(YOLO_CLASSES.keys()) - {15}
VIT_CLASS_NAMES = ["Cavity", "Fillings", "Impacted Tooth", "Implant", "Normal"]
SEGFORMER_MODEL_PATH = "SEGFORMER_MODEL"
VIT_MODEL_PATH = "VIT_MODEL"

class DentoVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DentalVision – Panoramic X-Ray Analysis")
        self.root.geometry("1300x900")

        api_key = "your_api_key_here" #Add Your API Key here
        if not api_key:
            messagebox.showerror("Missing API Key", "Set API_KEY environment variable")
            root.destroy()
            return
        self.client = genai.Client(api_key=api_key)

        try:
            # Models
            self.seg_processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_PATH)
            self.seg_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_PATH)
            
            self.vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_PATH)
            self.vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL_PATH)
            
            self.seg_model.eval()
            self.vit_model.eval()
            self.yolo = YOLO("YOLO_Model/Issues.pt")
            
            # Initialize the new YOLO-based Heatmap Generator
            self.heatmap_gen = YOLOHeatmapGenerator()
            
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load models: {e}")
            root.destroy()
            return

        self.current_image_path = None
        self.setup_ui()

    def setup_ui(self):
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(main, text="DentalVision – Clinical Diagnostic Assistant", font=("Arial", 18, "bold")).pack(anchor="w", pady=10)

        ctrl = tk.Frame(main)
        ctrl.pack(fill=tk.X)
        tk.Button(ctrl, text="📁 Select X-Ray", command=self.browse_image, bg="#2563eb", fg="white", padx=20, pady=10).pack(side=tk.LEFT, padx=5)
        self.btn_analyze = tk.Button(ctrl, text="🔍 Run Analysis", command=self.analyze_image, bg="#16a34a", fg="white", padx=20, pady=10, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT)
        tk.Button(ctrl, text="↺ Clear", command=self.reset_app, bg="#9ca3af", fg="white", padx=20, pady=10).pack(side=tk.RIGHT)

        content = tk.Frame(main)
        content.pack(fill=tk.BOTH, expand=True, pady=10)

        # LEFT PANEL: HEATMAP (YOLO Based)
        left = tk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.left_top = tk.Frame(left, bd=1, relief=tk.GROOVE)
        self.left_top.pack(fill=tk.BOTH, expand=True)
        tk.Label(self.left_top, text="Issue Heatmap (Red=Issue, Blue=Safe)", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
        self.vit_image_label = tk.Label(self.left_top)
        self.vit_image_label.pack(fill=tk.BOTH, expand=True)

        self.left_bottom = tk.Frame(left, bd=1, relief=tk.GROOVE)
        self.left_bottom.pack(fill=tk.X)
        tk.Label(self.left_bottom, text="Detection Summary", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
        self.health_label = tk.Label(self.left_bottom, text="🟢 Healthy Tooth: 0%")
        self.health_label.pack(anchor="w", padx=10)
        self.issue_label = tk.Label(self.left_bottom, text="🔴 Tooth with Issues: 0%")
        self.issue_label.pack(anchor="w", padx=10)
        tk.Label(self.left_bottom, text="Detected Issues:", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=(5, 0))
        self.issue_list = tk.Label(self.left_bottom, text="- None", justify=tk.LEFT)
        self.issue_list.pack(anchor="w", padx=20, pady=5)

        # RIGHT PANEL: YOLO BOXES
        right = tk.Frame(content)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.right_top = tk.Frame(right, bd=1, relief=tk.GROOVE)
        self.right_top.pack(fill=tk.BOTH, expand=True)
        tk.Label(self.right_top, text="YOLO Detection (Boxes)", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=5)
        self.yolo_image_label = tk.Label(self.right_top)
        self.yolo_image_label.pack(fill=tk.BOTH, expand=True)

        self.right_bottom = tk.Frame(right, bd=1, relief=tk.GROOVE)
        self.right_bottom.pack(fill=tk.X)
        tk.Label(self.right_bottom, text="AI Diagnostic Report", font=("Arial", 12, "bold")).pack(anchor="w", padx=10)
        self.results_text = scrolledtext.ScrolledText(self.right_bottom, height=8, wrap=tk.WORD)
        self.results_text.pack(fill=tk.X, padx=10, pady=5)
        self.results_text.config(state=tk.DISABLED)

    def browse_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.current_image_path = path
            img = Image.open(path)
            img.thumbnail((600, 600))
            photo = ImageTk.PhotoImage(img)
            self.vit_image_label.config(image=photo)
            self.vit_image_label.image = photo
            self.btn_analyze.config(state=tk.NORMAL)

    def process_vit_stats(self, image_path):
        """
        Runs Segformer + ViT purely for STATISTICS (counting healthy vs unhealthy).
        Does NOT generate visualization (visual is now handled by YOLO heatmap).
        """
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Segmentation
        inputs = self.seg_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.seg_model(**inputs)
        
        mask = torch.nn.functional.interpolate(
            outputs.logits, size=image_np.shape[:2], mode="bilinear", align_corners=False
        ).argmax(dim=1)[0].cpu().numpy()
        
        mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        vit_labels = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 20: continue
            
            tooth_crop = image_np[y:y+h, x:x+w]
            tooth_pil = Image.fromarray(tooth_crop)
            
            # Predict Label
            v_inputs = self.vit_processor(tooth_pil, return_tensors="pt")
            with torch.no_grad():
                v_outputs = self.vit_model(**v_inputs)
            label = VIT_CLASS_NAMES[v_outputs.logits.argmax(-1).item()]
            vit_labels.append(label)

        return vit_labels

    def analyze_image(self):
        # 1. Run YOLO (Used for Right Panel AND Left Panel Heatmap)
        results = self.yolo.predict(self.current_image_path, conf=0.4, imgsz=640)
        
        # --- LEFT PANEL: Generate Gradient Heatmap (No boxes, Blue background, Red issues) ---
        original_pil = Image.open(self.current_image_path).convert("RGB")
        heatmap_pil = self.heatmap_gen.generate_heatmap(original_pil, results, alpha=0.6)
        
        heatmap_pil.thumbnail((600, 600))
        self.vit_photo = ImageTk.PhotoImage(heatmap_pil)
        self.vit_image_label.config(image=self.vit_photo)
        # ------------------------------------------------------------------------------------

        # --- RIGHT PANEL: Standard YOLO Boxes ---
        yolo_annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        yolo_pil = Image.fromarray(yolo_annotated)
        yolo_pil.thumbnail((600, 600))
        self.yolo_photo = ImageTk.PhotoImage(yolo_pil)
        self.yolo_image_label.config(image=self.yolo_photo)

        # 2. Statistics (Using ViT Logic purely for data)
        # We run this in background just to get the Health % numbers
        vit_labels = self.process_vit_stats(self.current_image_path)

        detections = []
        for result in results:
            for box in result.boxes:
                detections.append(int(box.cls[0]))

        issue_names = sorted({YOLO_CLASSES[c] for c in detections if c in ISSUE_CLASSES_YOLO})
        self.issue_list.config(text="\n".join(f"- {i}" for i in issue_names) if issue_names else "- None")

        # Severity Logic
        total_teeth = len(vit_labels)
        if total_teeth == 0: total_teeth = 1 # Prevent div by zero if seg fails
        
        # Simple logic: If YOLO found issues, health is lower
        if len(issue_names) > 0:
             # Just an example logic, you can adjust
            i_pct = min(100.0, len(issue_names) * 15.0) 
        else:
            i_pct = 0.0
            
        h_pct = round(100.0 - i_pct, 1)
        self.health_label.config(text=f"🟢 Healthy Tooth: {h_pct}%")
        self.issue_label.config(text=f"🔴 Tooth with Issues: {i_pct}%")

        # AI PROMPT
        if issue_names:
            findings_text = ", ".join(issue_names)
            prompt = (
                f"A dental analysis has identified these specific issues: {findings_text}.\n\n"
                "Please generate a detailed report providing:\n"
                "1. Clinical details for each issue.\n"
                "2. Potential causes.\n"
                "3. Typical treatment paths.\n\n"
                "Disclaimer: AI-generated report. Consult a dentist."
            )
        else:
            prompt = (
                "The dental scan found no abnormalities.\n\n"
                "Provide tips for maintaining long-term oral hygiene."
            )

        # AI Call
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[prompt] 
                )
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, response.text)
                self.results_text.config(state=tk.DISABLED)
                break
            except Exception as e:
                if "503" in str(e) and attempt < 2:
                    time.sleep(2)
                    continue
                self.results_text.config(state=tk.NORMAL)
                self.results_text.insert(tk.END, f"\n[Report service unavailable. Error: {e}]")
                self.results_text.config(state=tk.DISABLED)
                break

    def reset_app(self):
        self.vit_image_label.config(image="")
        self.yolo_image_label.config(image="")
        self.issue_list.config(text="- None")
        self.health_label.config(text="🟢 Healthy Tooth: 0%")
        self.issue_label.config(text="🔴 Tooth with Issues: 0%")
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Analysis report will appear here.")
        self.results_text.config(state=tk.DISABLED)
        self.btn_analyze.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = DentoVisionApp(root)
    root.mainloop()