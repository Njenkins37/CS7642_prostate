import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import sys

from loader import load_picai_case, pos_list, IMAGES_ROOT, LABELS_ROOT

class CaseNavigator:
    def __init__(self, case_ids, img_root, lbl_root):
        """
        """
        self.case_ids = case_ids
        self.img_root = img_root
        self.lbl_root = lbl_root
        
        # Navigation State
        self.case_idx = 0
        self.current_case_id = None
        
        # Image Data State
        self.volumes = {}
        self.masks = {}
        self.volume_names = ["T2", "ADC"]
        self.current_vol_idx = 0  # 0 for T2, 1 for ADC
        self.show_masks = True
        
        # Slice State
        self.slice_idx = 0
        self.num_slices = 0

        # Colors
        self.colors = {
            "lesion": 'red',
            "PZ": 'cyan',
            "TZ": 'lime'
        }

        # Initialize Plot
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.im = None

        # Connect events
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Load the first case
        self.load_case(self.case_idx)
        
        print("\n=== CONTROLS ===")
        print(" [Scroll/Up/Down] : Change Slice")
        print(" [Left/Right]     : Toggle T2 / ADC")
        print(" [N]              : Next Case")
        print(" [B]              : Previous (Back) Case")
        print(" [M]              : Toggle Overlays")
        print("================\n")

        plt.show()

    def load_case(self, idx):
        """Loads data for the case at self.case_ids[idx]"""
        # Wrap index to ensure it stays valid
        self.case_idx = idx % len(self.case_ids)
        self.current_case_id = self.case_ids[self.case_idx]
        
        print(f"Loading Case: {self.current_case_id} ({self.case_idx + 1}/{len(self.case_ids)})...")

        try:
            # 1. Load Data using your loader.py function
            data = load_picai_case(self.current_case_id, self.img_root, self.lbl_root)

            # 2. Organize Volumes
            self.volumes = {
                "T2": data["t2"],
                "ADC": data["adc"]
            }

            # 3. Organize Masks (aligned to T2 geometry)
            self.masks = {
                "T2": {
                    "lesion": data["lesion_t2"],
                    "PZ": (data["zone_t2"] == 1).astype(np.uint8),
                    "TZ": (data["zone_t2"] == 2).astype(np.uint8)
                },
                "ADC": {
                    "lesion": data["lesion_adc"],
                    "PZ": (data["zone_adc"] == 1).astype(np.uint8),
                    "TZ": (data["zone_adc"] == 2).astype(np.uint8)
                }
            }

            # 4. Reset Slice to Middle
            self.num_slices = self.volumes["T2"].shape[2]
            self.slice_idx = self.num_slices // 2
            
            # 5. Refresh Display
            self.update_plot()

        except Exception as e:
            print(f"Error loading {self.current_case_id}: {e}")
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Error loading {self.current_case_id}\n{e}", 
                         ha='center', va='center', color='red')
            self.fig.canvas.draw_idle()

    def update_plot(self):
        """Redraws the image and contours"""
        self.ax.clear()

        # Get current volume and masks
        vol_name = self.volume_names[self.current_vol_idx]
        volume = self.volumes.get(vol_name)
        
        if volume is None:
            return

        # Display Image (Transpose X,Y -> Y,X for matplotlib)
        img_slice = volume[:, :, self.slice_idx].T
        self.ax.imshow(img_slice, cmap='gray', vmin=0, vmax=1, origin='upper')

        # Display Contours
        if self.show_masks:
            current_masks = self.masks.get(vol_name, {})
            for name, mask_vol in current_masks.items():
                if mask_vol is not None:
                    mask_slice = mask_vol[:, :, self.slice_idx].T
                    color = self.colors.get(name, 'red')
                    self._draw_single_contour(mask_slice, color)

        # Title / Info
        self.ax.set_title(
            f"Case: {self.current_case_id} | {vol_name} | Slice: {self.slice_idx}/{self.num_slices}"
        )
        self.ax.axis('off')
        self.fig.canvas.draw_idle()

    def _draw_single_contour(self, mask_slice, color):
        """Helper to draw contour using skimage"""
        contours = measure.find_contours(mask_slice, 0.5)
        for c in contours:
            # c is (row, col) -> plot (x, y) = (col, row)
            self.ax.plot(c[:, 1], c[:, 0], color=color, linewidth=1.5)

    def on_scroll(self, event):
        """Handle mouse scroll for slices"""
        if event.button == 'up':
            self.slice_idx = min(self.slice_idx + 1, self.num_slices - 1)
        elif event.button == 'down':
            self.slice_idx = max(self.slice_idx - 1, 0)
        self.update_plot()

    def on_key(self, event):
        """Handle keyboard inputs"""
        key = event.key.lower()

        # --- Slice Navigation ---
        if key == 'up':
            self.slice_idx = min(self.slice_idx + 1, self.num_slices - 1)
        elif key == 'down':
            self.slice_idx = max(self.slice_idx - 1, 0)
        
        # --- Modality Switching (T2 <-> ADC) ---
        elif key == 'right':
            self.current_vol_idx = (self.current_vol_idx + 1) % len(self.volume_names)
        elif key == 'left':
            self.current_vol_idx = (self.current_vol_idx - 1) % len(self.volume_names)
        
        # --- Case Navigation ---
        elif key == 'n':  # Next Case
            self.load_case(self.case_idx + 1)
            return # load_case calls update_plot
        elif key == 'b':  # Back / Previous Case
            self.load_case(self.case_idx - 1)
            return
            
        # --- Toggle Options ---
        elif key == 'm':
            self.show_masks = not self.show_masks
        
        self.update_plot()

if __name__ == "__main__":
    # Check if list is empty
    if not pos_list:
        print("Error: pos_list in loader.py is empty.")
    else:
        # Start the viewer with the lists defined in loader.py
        viewer = CaseNavigator(pos_list, IMAGES_ROOT, LABELS_ROOT)