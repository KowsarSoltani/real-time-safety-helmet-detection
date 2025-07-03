import cv2
import torch
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
import logging
import time
import os
import tkinter as tk
from tkinter import scrolledtext, ttk, simpledialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
from tkinter import font as tkfont
from datetime import datetime

# Configure logging
log_filename = f'webcam_processing_{time.strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def apply_canny_pytorch(img, low_threshold=50, high_threshold=150, weight=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_tensor = torch.from_numpy(img).float().to(device) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    gray_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(1, 3, 1, 1)
    gray = torch.sum(img_tensor * gray_weights, dim=1, keepdim=True)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-10)
    grad_direction = torch.atan2(grad_y, grad_x)
    grad_direction = grad_direction * 180 / torch.pi
    grad_direction = torch.round(grad_direction / 45) * 45
    suppressed = torch.zeros_like(grad_magnitude)
    _, _, H, W = grad_magnitude.shape
    for angle in [0, 45, 90, 135]:
        mask = (grad_direction == angle) | (grad_direction == angle - 180)
        padded = F.pad(grad_magnitude, (1, 1, 1, 1), mode='constant', value=0)
        if angle == 0:
            left = padded[:, :, 1:H+1, 0:W]
            right = padded[:, :, 1:H+1, 2:W+2]
            neighbors = torch.stack([left, right], dim=0)
        elif angle == 90:
            up = padded[:, :, 0:H, 1:W+1]
            down = padded[:, :, 2:H+2, 1:W+1]
            neighbors = torch.stack([up, down], dim=0)
        elif angle == 45:
            top_left = padded[:, :, 0:H, 0:W]
            bottom_right = padded[:, :, 2:H+2, 2:W+2]
            neighbors = torch.stack([top_left, bottom_right], dim=0)
        else:  # 135
            top_right = padded[:, :, 0:H, 2:W+2]
            bottom_left = padded[:, :, 2:H+2, 0:W]
            neighbors = torch.stack([top_right, bottom_left], dim=0)
        max_neighbor = torch.max(neighbors, dim=0)[0]
        suppressed[mask] = grad_magnitude[mask] * (grad_magnitude[mask] >= max_neighbor[mask]).float()
    strong_edges = (suppressed > high_threshold / 255.0).float()
    weak_edges = ((suppressed >= low_threshold / 255.0) & (suppressed <= high_threshold / 255.0)).float()
    kernel = torch.ones(1, 1, 3, 3, device=device)
    strong_dilated = F.conv2d(strong_edges, kernel, padding=1) > 0
    edges = strong_edges + weak_edges * strong_dilated.float()
    edges = edges.clamp(0, 1)
    edges_colored = edges.repeat(1, 3, 1, 1)
    enhanced = (1.0 - weight) * img_tensor + weight * edges_colored
    enhanced = enhanced.clamp(0, 1) * 255.0
    enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return enhanced

def apply_helmet_detection(frame, person_model, helmet_model, show_boxes=True):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect people first
    person_results = person_model(img_rgb, conf=0.3, iou=0.5, classes=[0])  # class 0 is person
    
    # Convert to HSV for color detection
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for yellow and red hard hats
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    
    # Create masks for yellow and red colors
    yellow_mask = cv2.inRange(img_hsv, yellow_lower, yellow_upper)
    red_mask1 = cv2.inRange(img_hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(img_hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Combine masks
    helmet_mask = cv2.bitwise_or(yellow_mask, red_mask)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5,5), np.uint8)
    helmet_mask = cv2.morphologyEx(helmet_mask, cv2.MORPH_OPEN, kernel)
    helmet_mask = cv2.morphologyEx(helmet_mask, cv2.MORPH_CLOSE, kernel)
    
    output_frame = frame.copy()
    helmet_count = 0
    
    if show_boxes and len(person_results) > 0:
        # Get person detections
        person_boxes = person_results[0].boxes.xyxy.cpu().numpy()
        
        # Process each person detection
        for person_box in person_boxes:
            x1, y1, x2, y2 = map(int, person_box)
            person_width = x2 - x1
            person_height = y2 - y1
            
            # Define head region more precisely (top 1/6 to 1/4 of person)
            head_y1 = int(y1 + person_height * 0.05)  # Start a bit below top to avoid false positives
            head_y2 = int(y1 + person_height * 0.22)  # Typical head region
            head_x1 = int(x1 + person_width * 0.2)    # Narrow the width for more precise detection
            head_x2 = int(x2 - person_width * 0.2)
            
            # Extract head region
            head_region = helmet_mask[head_y1:head_y2, head_x1:head_x2]
            
            # Skip if head region is too small
            if head_region.shape[0] * head_region.shape[1] == 0:
                continue
                
            # Find contours in the head region
            contours, _ = cv2.findContours(head_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            has_helmet = False
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                if area < 50:  # Filter out small noise
                    continue
                    
                # Get the bounding rectangle of the contour
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate aspect ratio and area ratio
                aspect_ratio = float(w) / h if h > 0 else 0
                area_ratio = float(area) / (head_region.shape[0] * head_region.shape[1])
                
                # Check if the contour matches helmet properties
                if (0.8 <= aspect_ratio <= 2.0 and  # Typical helmet aspect ratio
                    0.3 <= area_ratio <= 0.85 and   # Helmet should cover significant but not all of head region
                    y <= head_region.shape[0] * 0.5):  # Helmet should be in upper part of head region
                    
                    has_helmet = True
                    break
            
            if has_helmet:
                # Draw head region in blue for debugging
                cv2.rectangle(output_frame, (head_x1, head_y1), (head_x2, head_y2), (255, 0, 0), 1)
                
                # Green box for person with helmet
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_frame, "HAT", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                helmet_count += 1
            else:
                # Draw head region in blue for debugging
                cv2.rectangle(output_frame, (head_x1, head_y1), (head_x2, head_y2), (255, 0, 0), 1)
                
                # Red box for person without helmet
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return output_frame, helmet_count

def save_screenshot(frame, helmet_count, timestamp):
    screenshots_dir = os.path.abspath('screenshots')
    os.makedirs(screenshots_dir, exist_ok=True)

    text = f"People with Helmets: {helmet_count}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 50), 2)

    try:
        dt_obj = datetime.strptime(timestamp, "%Y%m%d%H%M%S")
        timestamp_formatted = dt_obj.strftime("%Y%m%d_%H%M%S")
    except ValueError:
        timestamp_formatted = timestamp

    filename = f"screenshot_{timestamp_formatted}.png"
    filepath = os.path.join(screenshots_dir, filename)

    success = cv2.imwrite(filepath, frame)
    if success:
        logging.info(f"Screenshot saved: {filepath} with {helmet_count} people wearing helmets")
        return {"filename": filename, "helmet_count": helmet_count}
    else:
        logging.error(f"Error saving screenshot: {filepath}")
        return None

def show_log_and_screenshots(root, log_filename, current_screenshots):
    log_window = tk.Toplevel(root)
    log_window.title("Logs & Screenshot Gallery")
    log_window.geometry("900x650")
    log_window.configure(bg="#1e1e2e")

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TLabel", font=("Segoe UI", 14, "bold"), background="#1e1e2e", foreground="#ffffff")
    style.configure("TButton", font=("Segoe UI", 12), padding=10, background="#3b82f6", foreground="#ffffff")
    style.map("TButton", background=[('active', '#2563eb')])
    style.configure("TFrame", background="#1e1e2e")

    # Log Section
    log_frame = ttk.Frame(log_window)
    log_frame.pack(padx=20, pady=20, fill="both", expand=True)

    ttk.Label(log_frame, text="Log Messages", style="TLabel").pack(anchor="w", pady=(0, 10))
    log_text = scrolledtext.ScrolledText(log_frame, width=100, height=10, font=("Segoe UI", 10),
                                         bg="#2a2a3a", fg="#ffffff", insertbackground="#ffffff")
    log_text.pack(fill="both", expand=True)

    try:
        with open(log_filename, 'r') as log_file:
            log_lines = log_file.readlines()
            log_content = ''.join(log_lines[::-1][:500])
            log_text.insert(tk.END, log_content)
        log_text.config(state='disabled')
    except FileNotFoundError:
        log_text.insert(tk.END, "Log file not found.")
        log_text.config(state='disabled')

    # Screenshot Gallery
    screenshot_frame = ttk.Frame(log_window)
    screenshot_frame.pack(padx=20, pady=20, fill="both", expand=True)

    ttk.Label(screenshot_frame, text="Screenshot Gallery", style="TLabel").pack(anchor="w", pady=(0, 10))
    canvas = tk.Canvas(screenshot_frame, bg="#1e1e2e", highlightthickness=0)
    scrollbar = ttk.Scrollbar(screenshot_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    photo_references = []
    screenshots_dir = os.path.abspath('screenshots')
    screenshot_files = sorted(current_screenshots, key=lambda x: x["filename"], reverse=True)[:12]
    logging.info(f"Number of screenshots found for current session: {len(screenshot_files)}")

    if not screenshot_files:
        ttk.Label(scrollable_frame, text="No screenshots available.", font=("Segoe UI", 12),
                  foreground="#ffffff").pack(pady=20)
    else:
        for i, screenshot in enumerate(screenshot_files):
            filename = screenshot["filename"]
            helmet_count = screenshot["helmet_count"]

            try:
                timestamp_raw = filename.split('_')[1].replace('.png', '')
                dt_obj = datetime.strptime(timestamp_raw, "%Y%m%d_%H%M%S")
                timestamp = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                timestamp = "Invalid timestamp"

            try:
                img_path = os.path.join(screenshots_dir, filename)
                logging.info(f"Loading screenshot: {img_path}")
                img = Image.open(img_path)
                img = img.resize((200, 150), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                photo_references.append(photo)

                entry_frame = ttk.Frame(scrollable_frame, style="TFrame")
                entry_frame.grid(row=i // 3, column=i % 3, padx=10, pady=10)

                img_label = tk.Label(entry_frame, image=photo, bg="#2a2a3a", bd=2, relief="flat")
                img_label.image = photo
                img_label.pack()

                img_label.bind("<Enter>", lambda e, lbl=img_label: lbl.configure(relief="raised"))
                img_label.bind("<Leave>", lambda e, lbl=img_label: lbl.configure(relief="flat"))

                details = f"Helmets: {helmet_count}\nTime: {timestamp}"
                ttk.Label(entry_frame, text=details, font=("Segoe UI", 10), foreground="#ffffff",
                          justify="center").pack(pady=5)
            except Exception as e:
                logging.error(f"Error loading screenshot {filename}: {str(e)}")
                ttk.Label(scrollable_frame, text=f"Error loading {filename}",
                          font=("Segoe UI", 10), foreground="#ffffff").grid(row=i // 3, column=i % 3, padx=10, pady=10)

class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Safety Helmet Detection")
        self.root.geometry("700x600")
        self.root.configure(bg="#1e1e2e")

        self.header_font = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        self.button_font = tkfont.Font(family="Segoe UI", size=12)

        self.current_screenshots = []
        self.log_filename = log_filename
        
        # Load both models - one for person detection and one for hard hat detection
        self.person_model = YOLO('yolov8n.pt')  # For person detection
        self.helmet_model = YOLO('yolov8n.pt')  # Temporarily using same model, we'll update when downloaded
        
        # Configure the helmet model to detect construction helmets more accurately
        self.person_model.conf = 0.3  # Lower confidence threshold for persons
        self.helmet_model.conf = 0.25  # Lower confidence threshold for helmets
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Initialized YOLO models on device: {self.device}")

        self.cap = None
        self.last_screenshot_time = time.time()
        self.screenshot_interval = 5
        self.running = True
        self.photo_references = []
        self.show_boxes = True

        # Styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=self.button_font, padding=10, background="#3b82f6", foreground="#ffffff")
        style.map("TButton", background=[('active', '#2563eb')])
        style.configure("TLabel", font=("Segoe UI", 12), background="#1e1e2e", foreground="#ffffff")

        # Main Frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Header
        ttk.Label(self.main_frame, text="Safety Helmet Detection System", font=self.header_font, foreground="#ffffff").pack(pady=(0, 20))

        # Video Feed Frame
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(fill="both", expand=True)

        # Create placeholder for video feed
        placeholder = Image.new('RGB', (640, 480), color=(50, 50, 70))
        draw = ImageDraw.Draw(placeholder)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((20, 200), "Initializing Camera...", fill=(255, 255, 255), font=font)
        photo = ImageTk.PhotoImage(placeholder)
        
        self.video_label = tk.Label(self.video_frame, image=photo, bg="#2a2a3a", bd=2, relief="flat")
        self.video_label.image = photo
        self.video_label.pack(pady=10)

        # Control Buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(pady=20)
        
        self.toggle_btn = ttk.Button(button_frame, text="Hide Boxes", 
                                   command=self.toggle_boxes)
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="View Logs & Gallery", 
                  command=lambda: show_log_and_screenshots(self.root, self.log_filename, self.current_screenshots)).pack(side=tk.LEFT, padx=5)

        # Start webcam
        self.start_webcam()

        # Start video processing thread
        self.process_thread = threading.Thread(target=self.process_video)
        self.process_thread.daemon = True
        self.process_thread.start()

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)  # Use default webcam
        if not self.cap.isOpened():
            logging.error("Failed to open webcam")
            messagebox.showerror("Error", "Failed to open webcam")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        logging.info("Webcam initialized successfully")

    def process_video(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to capture frame from webcam")
                continue

            canny_output = apply_canny_pytorch(frame, low_threshold=50, high_threshold=150, weight=0.2)
            output_with_boxes, helmet_count = apply_helmet_detection(canny_output, self.person_model, self.helmet_model, self.show_boxes)
            
            current_time = time.time()
            if current_time - self.last_screenshot_time >= self.screenshot_interval:
                timestamp = time.strftime("%Y%m%d%H%M%S")
                screenshot_info = save_screenshot(output_with_boxes, helmet_count, timestamp)
                if screenshot_info:
                    self.current_screenshots.append(screenshot_info)
                self.last_screenshot_time = current_time

            frame_rgb = cv2.cvtColor(output_with_boxes, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(img)
            
            self.root.after(0, self.update_video_label, photo)
            time.sleep(0.033)

    def update_video_label(self, photo):
        self.video_label.configure(image=photo)
        self.video_label.image = photo
        self.photo_references.append(photo)

    def toggle_boxes(self):
        self.show_boxes = not self.show_boxes
        self.toggle_btn.configure(text="Show Boxes" if not self.show_boxes else "Hide Boxes")
        logging.info(f"Bounding boxes {'hidden' if not self.show_boxes else 'shown'}")

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.quit()

def main():
    root = tk.Tk()
    app = VideoApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()
    logging.info("Application terminated")

if __name__ == '__main__':
    main()