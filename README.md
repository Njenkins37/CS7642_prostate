# CS7642 Prostate Cancer Detection

This repository serves as the repo for the final project of CS7642. 

## 1. Getting Started
1. Clone the repository locally
2. run `python -m venv .venv`
3. Make the new .venv your interpreter for your IDE
4. Run `source .venv/bin/activate`  on mac or `venv\Scripts\activate` on PC
5. Run `pip install -r requirements.txt`

---

## 2. Preprocessing Pipeline (`preprocess_dataset.py`)
To solve the spatial variance in MRI scans, we center the prostate and extracts a normalized $128 \times 128$ tensor across T2 and ADC modalities.

### Key Design Decisions:
* **Anatomical Centering:** We use the AI-generated Whole Gland mask (**Bosma22b**) to calculate the Center of Mass (CoM). This ensures the crop is centered on the organ, not just the tumor, providing the model with consistent anatomical context.
* **128x128 Resolution:** At ~0.5mm/pixel, this crop captures ~64mm of tissue, which hopefully safely encapsulates the average prostate (~40-50mm) plus a healthy margin.
Now it accepts variable crop size via `crop_size` parameter.
* **Registration:** ADC images (lower resolution) are resampled into the T2 reference space before cropping to ensure 1:1 pixel alignment for the Cross-Attention bottleneck.

### Crop Strategies (`--strategy`):
We provide two modes to facilitate ablation studies on spatial alignment:
1.  **Strict (Default):** Rejects any patient where the 128x128 box clips the tumor. 
    * *Use case:* When absolute anatomical centering is required.
2.  **Shift:** If a tumor is clipped by the bounding box, the script recalculates the CoM based on tumor to draw another bounding box containing it.
    * *Use case:* Maximizing training data (recovers ~50+ patients) and testing the model's robustness to off-center anatomy.

**To run:**
```bash
python preprocess_dataset.py --strategy strict --crop_size 128
```

The code also generates logs and json files that contain the results for reuse.