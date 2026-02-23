# DentoVision-AI


### Panoramic Dental X-Ray Analysis using YOLOv8 + SegFormer + ViT + Gemini AI

------------------------------------------------------------------------

## 🚀 Overview

DentoVision-AI is an AI-powered clinical diagnostic assistant for
analyzing panoramic dental X-rays.

It integrates: - YOLOv8 for multi-class dental issue detection -
SegFormer for tooth segmentation - Vision Transformer (ViT) for
tooth-level classification - Google Gemini for AI-generated clinical
reporting - Tkinter for a desktop-based graphical interface

The system provides: - Tooth health percentage estimation - Detected
dental abnormalities - Annotated X-ray output - AI-generated structured
clinical report - Automatic saving of analysis results

------------------------------------------------------------------------

## 🏗 System Architecture

### Processing Pipeline

1.  User uploads panoramic X-ray image
2.  SegFormer performs semantic tooth segmentation
3.  Extracted tooth regions are classified using ViT
4.  YOLO detects dental abnormalities
5.  Gemini generates detailed clinical explanation
6.  Results are displayed and saved automatically

------------------------------------------------------------------------

## 📂 Repository Structure

```
DentoVision-AI/
│
├── main.py
│
├── YOLO_Model/
│   └── Issues.pt
│
├── VIT_MODEL/
│   ├── config.json
│   ├── model.safetensors
│   ├── training_args.bin
│   └── preprocessor_config.json
│
├── SEGFORMER_MODEL/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
│
├── outputs/
│
└── README.md
```
------------------------------------------------------------------------

## 🧠 Models Used

### YOLOv8 -- Dental Issue Detection

-   Framework: Ultralytics
-   Confidence Threshold: 0.4
-   Multi-class detection (31 dental categories)

### SegFormer -- Tooth Segmentation

-   Framework: HuggingFace Transformers
-   Mask-based contour extraction for individual teeth

### Vision Transformer (ViT) -- Tooth Classification

Classes: - Cavity - Fillings - Impacted Tooth - Implant - Normal

### Gemini AI -- Clinical Report Generation

-   Model: gemini-3-flash-preview
-   Generates:
    -   Clinical interpretation
    -   Potential causes
    -   Treatment pathways
    -   Professional disclaimer

------------------------------------------------------------------------

## 🖥 GUI Features

-   Upload X-ray image
-   Run complete AI analysis
-   View annotated YOLO output
-   Tooth health vs issue percentage
-   AI-generated structured report
-   Auto-save image + text report

------------------------------------------------------------------------

## 💾 Output Files

All results are saved inside:

outputs/

Generated files: - yolo_analysis.jpg - analysis_report.txt

------------------------------------------------------------------------

## ⚙️ Installation Guide

### 1. Clone Repository

git clone https://github.com/MidhnM/DentoVision-AI.git cd
DentoVision-AI

### 2. Create Virtual Environment

python -m venv venv venv`\Scripts`{=tex}`\activate`{=tex}

### 3. Install Dependencies

pip install torch torchvision transformers ultralytics opencv-python
pillow numpy google-generativeai

### 4. Set Environment Variable for Gemini

set GEMINI_API_KEY=your_api_key_here

### 5. Run Application

python app.py

------------------------------------------------------------------------

## 📊 Severity Logic

Health percentage is estimated based on detected tooth count ranges and
abnormal findings.

------------------------------------------------------------------------

## 🔐 Security Recommendation

Do NOT hardcode API keys inside source files.\
Use environment variables instead.

------------------------------------------------------------------------

## 📌 Future Improvements

-   REST API deployment
-   Model quantization for edge devices
-   Performance benchmarking (mAP, IoU, F1)
-   Cloud deployment (Streamlit / Flask)
-   Extended dental dataset training

------------------------------------------------------------------------

## ⚠️ Medical Disclaimer

This project is intended for research and educational purposes only.\
It is NOT a replacement for professional dental diagnosis.\
Always consult a licensed dental professional.

------------------------------------------------------------------------

## 👨‍💻 Author

Midhun M\
Electronics Engineer | RF Engineer\
AI & Medical Imaging Enthusiast

------------------------------------------------------------------------

## ⭐ Support

If you find this project useful: - Star the repository - Fork it -
Contribute improvements
