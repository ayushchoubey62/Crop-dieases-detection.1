import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps
import time
import threading
import os
from fpdf import FPDF
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import sys
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='crop_diagnosis.log')

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Initialize models
        self.disease_model = None
        self.crop_classifier = None
        self.class_labels = None

        self.frames = {}
        
        try:
            # Load the model with error handling
            self.load_models()
            # Define class labels - make sure these match your model's output
            self.class_labels = {
                'Apple_Apple_scab': 0, 'Apple_Black_rot': 1, 'Apple_Cedar_apple_rust': 2, 
                'Apple_healthy': 3, 'Blueberry_healthy': 4, 'Cherry_Powdery_mildew': 5, 
                'Cherry_healthy': 6, 'Corn_Cercospora_leaf_spot_Gray_leaf_spot': 7, 
                'Corn_Common_rust': 8, 'Corn_Northern_Leaf_Blight': 9, 'Corn_healthy': 10, 
                'Grape_Black_rot': 11, 'Grape_Esca_Black_Measles': 12, 'Grape_Leaf_blight': 13, 
                'Grape_healthy': 14, 'Orange_Haunglongbing_Citrus_greening': 15, 
                'Peach_Bacterial_spot': 16, 'Peach_healthy': 17, 'Pepper_bell_Bacterial_spot': 18, 
                'Pepper_bell_healthy': 19, 'Potato_Early_blight': 20, 'Potato_Late_blight': 21, 
                'Potato_healthy': 22, 'Raspberry_healthy': 23, 'Soybean_healthy': 24, 
                'Squash_Powdery_mildew': 25, 'Strawberry_Leaf_scorch': 26, 'Strawberry_healthy': 27, 
                'Tomato_Bacterial_spot': 28, 'Tomato_Early_blight': 29, 'Tomato_Late_blight': 30, 
                'Tomato_Leaf_Mold': 31, 'Tomato_Septoria_leaf_spot': 32, 
                'Tomato_Spider_mites _Two_spotted_spider_mite': 33, 'Tomato_Target_Spot': 34, 
                'Tomato_Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato_Tomato_mosaic_virus': 36, 
                'Tomato_healthy': 37
            }
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            messagebox.showerror("Error", "Failed to load the disease detection model. The application may not function properly.")

        # Initialize UI
        self.title("Crop Disease Detection")
        self.geometry("1000x700")
        self.minsize(800, 600)
        
        self.recent_images = []
        self.diagnosis_history = []
        self.dark_mode = False
        self.icon_images = {}

        self.create_styles()
        self.load_icons()

        # Create container for all frames
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        container.grid_rowconfigure(0, weight=1)  
        container.grid_columnconfigure(0, weight=1)

        # Create all frames
        for F in (HomePage, DiagnosisPage, HistoryPage, FeedbackPage, HelpPage):
            page_name = F.__name__
            if page_name == "HelpPage":
                frame = F(parent=container, controller=self, icon_images=self.icon_images)
            else:
                frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.show_frame("HomePage")

    def create_styles(self):
        """Create modern UI styles"""
        self.style = ttk.Style()
        
        # Light mode styles
        self.style.configure('.', font=('Helvetica', 12), background='#F5F5F5', foreground='#212121')
        
        self.style.configure('TButton',
                           background='#1E88E5',
                           foreground='white',
                           padding=10,
                           borderwidth=0)
        self.style.map('TButton',
                      background=[('active', '#1976D2'), ('disabled', '#BBDEFB')],
                      foreground=[('disabled', '#757575')])
        
        self.style.configure('TFrame', background='#F5F5F5')
        self.style.configure('TLabel', background='#F5F5F5', foreground='#212121')
        self.style.configure('TLabelframe', background='#F5F5F5')
        self.style.configure('TLabelframe.Label', background='#F5F5F5', foreground='#212121')
        self.style.configure('TEntry', fieldbackground='white', foreground='#212121')
        self.style.configure('TCombobox', fieldbackground='white', foreground='#212121')
        self.style.configure('TProgressbar', background='#1E88E5', troughcolor='#E0E0E0')

    def load_models(self):
        """Load ML models with error handling"""
        model_files = {
            "disease_model": "my_model.keras",
            #"crop_classifier": "crop_classifier.keras"  # optional
        }
        
        for model_name, filename in model_files.items():
            try:
                model_path = self.get_resource_path(f"models/{filename}")
                if not os.path.exists(model_path):
                    logging.warning(f"Model file not found: {model_path}")
                    continue
                    
                setattr(self, model_name, load_model(model_path))
                logging.info(f"Loaded {model_name} successfully")
                
            except Exception as e:
                logging.error(f"Failed to load {model_name}: {e}")
                if model_name == "disease_model":
                    raise RuntimeError(f"Critical model failed to load: {e}")

    def get_resource_path(self, relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        
        full_path = os.path.join(base_path, relative_path)
        logging.info(f"Resource path resolved: {full_path}")
        return full_path

    def load_icons(self):
        """Load all icons with error handling"""
        icon_names = {
            "home_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 234500.png",
            "upload_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 234602.png",
            "history_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 234709.png",
            "feedback_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 234802.png",
            "help_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 235038.png",
            "dark_mode_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 235215.png",
            "quit_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 235257.png",
            "save_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 235406.png",
            "view_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 235500.png",
            "diagnose_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 235616.png",
            "delete_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 235700.png",
            "export_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 235742.png",
            "submit_icon": r"C:\Users\ayush\OneDrive\Desktop\project 1\icon image\Screenshot 2025-04-05 235844.png"
        }

        self.icon_images = {}
        for name, filename in icon_names.items():
            try:
                # Try multiple possible locations
                possible_paths = [
                    os.path.join("icons", filename),
                    self.get_resource_path(f"icons/{filename}"),
                    os.path.join(os.path.dirname(__file__), "icons", filename)
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        img = Image.open(path).resize((24, 24))
                        self.icon_images[name] = ImageTk.PhotoImage(img)
                        break
                else:
                    logging.warning(f"Icon not found: {filename}")
                    self.icon_images[name] = None
                    
            except Exception as e:
                logging.error(f"Error loading icon {filename}: {e}")
                self.icon_images[name] = None


    def toggle_dark_mode(self):
        """Toggle between light and dark mode"""
        self.dark_mode = not self.dark_mode
        
        if self.dark_mode:
            # Dark mode colors
            bg_color = "#212121"
            fg_color = "#EEEEEE"
            button_bg = "#424242"
            button_fg = "#FFFFFF"
            hover_bg = "#616161"
            disabled_bg = "#757575"
            entry_bg = "#424242"
        else:
            # Light mode colors
            bg_color = "#F5F5F5"
            fg_color = "#212121"
            button_bg = "#1E88E5"
            button_fg = "#FFFFFF"
            hover_bg = "#1976D2"
            disabled_bg = "#BBDEFB"
            entry_bg = "white"

        
        # Update style
        self.style.configure('.', background=bg_color, foreground=fg_color, fieldbackground=entry_bg)
        self.style.configure('TButton',
                           background=button_bg,
                           foreground=button_fg)
        self.style.map('TButton',
                      background=[('active', hover_bg), ('disabled', disabled_bg)])
        
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color, foreground=fg_color)
        self.style.configure('TLabelframe', background=bg_color)
        self.style.configure('TLabelframe.Label', background=bg_color, foreground=fg_color)
        self.style.configure('TEntry', fieldbackground=entry_bg, foreground=fg_color)
        self.style.configure('TCombobox', fieldbackground=entry_bg, foreground=fg_color)
        
        # Update all widgets
        for frame in self.frames.values():
            if isinstance(frame, tk.Frame):
                frame.configure(bg=bg_color)  # Only for regular tk.Frame
            self.update_widget_colors(frame, bg_color, fg_color)

    def update_widget_colors(self, widget, bg_color, fg_color):
        """Recursively update widget colors"""
        if isinstance(widget, tk.Frame):
            widget.configure(bg=bg_color)
        elif isinstance(widget, tk.Label):
            widget.configure(bg=bg_color, fg=fg_color)
        elif isinstance(widget, (tk.Text, tk.Listbox)):
            widget.configure(bg="#424242" if self.dark_mode else "#FFFFFF", 
                        fg=fg_color, 
                        insertbackground=fg_color)
    
        # Recursively update children
        for child in widget.winfo_children():
            self.update_widget_colors(child, bg_color, fg_color)

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()
        if hasattr(frame, "on_show"):
            frame.on_show()

    def quit(self):
        self.destroy()

    # Improved non-crop detection methods
    def is_non_crop_image(self, img_path):
        """
        Detect non-crop images using either:
        1. ML classifier (if available) - recommended
        2. Enhanced heuristic approach - fallback
        """
        if self.crop_classifier:
            return self._ml_non_crop_detection(img_path)
        else:
            return self._heuristic_non_crop_detection(img_path)

    def _ml_non_crop_detection(self, img_path):
        """Use trained classifier to detect non-crop images"""
        try:
            # Load and preprocess image
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict (assuming model outputs 0 for non-crop, 1 for crop)
            prediction = self.crop_classifier.predict(img_array)
            return prediction[0][0] < 0.5  # Threshold can be adjusted
            
        except Exception as e:
            logging.error(f"ML non-crop detection failed: {e}")
            return False  # Default to assuming it's a crop

    def _heuristic_non_crop_detection(self, img_path):
        """Enhanced heuristic approach for non-crop detection"""
        try:
            # Open image and convert to numpy array
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # 1. Check green color dominance (crop leaves are mostly green)
            green_pixels = np.sum((hsv[:,:,0] > 30) & (hsv[:,:,0] < 90))
            green_ratio = green_pixels / (img.size[0] * img.size[1])
            
            # 2. Check texture (crop leaves have more texture)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 3. Check brightness/contrast
            avg_brightness = np.mean(gray)
            contrast = gray.std()
            
            # Decision thresholds (tuned empirically)
            if (green_ratio < 0.3 or      # Not enough green
                laplacian_var < 100 or    # Too smooth
                avg_brightness < 30 or    # Too dark
                avg_brightness > 220 or   # Too bright
                contrast < 40):           # Low contrast
                return True
                
            return False
            
        except Exception as e:
            logging.error(f"Heuristic non-crop detection failed: {e}")
            return False  # Default to assuming it's a crop


class HomePage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller

        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Main container
        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Title label
        title_label = ttk.Label(
            main_frame, 
            text="Welcome to Crop Disease Diagnosis", 
            font=("Helvetica", 20, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 30), padx=10)

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # Navigation buttons
        buttons = [
            ("Diagnose Crop", "upload_icon", "DiagnosisPage"),
            ("View History", "history_icon", "HistoryPage"),
            ("Provide Feedback", "feedback_icon", "FeedbackPage"),
            ("Help & Guide", "help_icon", "HelpPage"),
            ("Toggle Dark Mode", "dark_mode_icon", None, lambda: controller.toggle_dark_mode()),
            ("Exit Application", "quit_icon", None, lambda: controller.quit())
        ]

        for i, (text, icon_name, page_name, *command) in enumerate(buttons):
            btn_frame = ttk.Frame(button_frame)
            btn_frame.grid(row=i, column=0, pady=5, padx=5, sticky="ew")
            btn_frame.grid_columnconfigure(0, weight=1)
            
            btn_command = command[0] if command else (lambda p=page_name: controller.show_frame(p))
            icon = controller.icon_images.get(icon_name)
            
            btn = ttk.Button(
                btn_frame,
                text=text,
                image=icon,
                compound=tk.LEFT,
                command=btn_command,
                style='TButton'
            )
            btn.grid(sticky="ew", padx=5, pady=5)
            
            # Add hover effect
            btn.bind("<Enter>", lambda e, b=btn: b.config(style='Hover.TButton'))
            btn.bind("<Leave>", lambda e, b=btn: b.config(style='TButton'))

        

class DiagnosisPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller
        self.uploaded_image = None
        self.current_image_path = None
        self.filtered_image = None

        # Configure grid weights
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Title label
        title_label = ttk.Label(self, text="Upload Crop Image for Diagnosis", 
                             font=("Helvetica", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20), padx=10)

        # Upload button
        upload_button = ttk.Button(
            self,
            text="Upload Crop Image",
            image=controller.icon_images.get("upload_icon"),
            compound=tk.LEFT,
            command=self.upload_image,
            style='TButton'
        )
        upload_button.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        upload_button.bind("<Enter>", lambda e: upload_button.config(style='Hover.TButton'))
        upload_button.bind("<Leave>", lambda e: upload_button.config(style='TButton'))

        # Image display frame
        self.image_frame = ttk.LabelFrame(self, text="Image Preview")
        self.image_frame.grid(row=2, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(self, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=3, column=0, columnspan=2, pady=10, padx=10)
        self.progress.grid_remove()

        self.status_label = ttk.Label(self, text="Ready to upload image", font=("Helvetica", 10))
        self.status_label.grid(row=4, column=0, columnspan=2, pady=5, padx=10)

        # Result label
        result_frame = ttk.LabelFrame(self, text="Diagnosis Results")
        result_frame.grid(row=5, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        
        self.result_label = ttk.Label(result_frame, text="No diagnosis available", 
                                   font=("Helvetica", 12), wraplength=600)
        self.result_label.pack(padx=10, pady=10)

        # Action buttons
        button_frame = ttk.Frame(self)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10, padx=10, sticky="ew")

        # Save button
        self.save_button = ttk.Button(
            button_frame,
            text="Save Report",
            image=controller.icon_images.get("save_icon"),
            compound=tk.LEFT,
            state=tk.DISABLED,
            command=self.save_report,
            style='TButton'
        )
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        self.save_button.bind("<Enter>", lambda e: self.save_button.config(style='Hover.TButton'))
        self.save_button.bind("<Leave>", lambda e: self.save_button.config(style='TButton'))

        back_button = ttk.Button(
            button_frame,
            text="Back to Home",
            command=lambda: controller.show_frame("HomePage"),
            style='TButton'
        )
        back_button.pack(side=tk.LEFT, padx=5, pady=5, expand=True, fill=tk.X)
        back_button.bind("<Enter>", lambda e: back_button.config(style='Hover.TButton'))
        back_button.bind("<Leave>", lambda e: back_button.config(style='TButton'))

        self.create_filter_controls()

    def create_filter_controls(self):
        """Create image filter controls"""
        filter_frame = ttk.LabelFrame(self, text="Image Adjustments")
        filter_frame.grid(row=7, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        
        # Brightness control
        ttk.Label(filter_frame, text="Brightness:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.brightness_scale = ttk.Scale(
            filter_frame,
            from_=0.1,
            to=2.0,
            value=1.0,
            command=lambda v: self.adjust_brightness(float(v))
        )
        self.brightness_scale.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        # Contrast control
        ttk.Label(filter_frame, text="Contrast:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.contrast_scale = ttk.Scale(
            filter_frame,
            from_=0.1,
            to=2.0,
            value=1.0,
            command=lambda v: self.adjust_contrast(float(v))
        )
        self.contrast_scale.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        
        # Saturation control
        ttk.Label(filter_frame, text="Saturation:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.saturation_scale = ttk.Scale(
            filter_frame,
            from_=0.0,
            to=2.0,
            value=1.0,
            command=lambda v: self.adjust_saturation(float(v))
        )
        self.saturation_scale.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        
        # Sharpen button
        sharpen_button = ttk.Button(
            filter_frame,
            text="Sharpen",
            image=self.controller.icon_images.get("sharpen_icon"),
            compound=tk.LEFT,
            command=self.sharpen_image,
            style='TButton'
        )
        sharpen_button.grid(row=3, column=0, columnspan=2, pady=5, padx=5, sticky="ew")
        sharpen_button.bind("<Enter>", lambda e: sharpen_button.config(style='Hover.TButton'))
        sharpen_button.bind("<Leave>", lambda e: sharpen_button.config(style='TButton'))
        
        # Reset button
        reset_button = ttk.Button(
            filter_frame,
            text="Reset Filters",
            command=self.reset_filters,
            style='TButton'
        )
        reset_button.grid(row=4, column=0, columnspan=2, pady=5, padx=5, sticky="ew")
        reset_button.bind("<Enter>", lambda e: reset_button.config(style='Hover.TButton'))
        reset_button.bind("<Leave>", lambda e: reset_button.config(style='TButton'))
        
        # Configure columns
        filter_frame.grid_columnconfigure(1, weight=1)

    def upload_image(self):
        """Handle image upload"""
        file_types = [
            ("Image files", "*.jpg;*.jpeg;*.png;*.bmp"),
            ("All files", "*.*")
        ]
        file_path = filedialog.askopenfilename(
            title="Select Crop Image",
            filetypes=file_types
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.uploaded_image = Image.open(file_path)
                self.filtered_image = self.uploaded_image.copy()
                self.display_image(self.uploaded_image)
                
                # Start processing
                self.start_processing(file_path)
                
            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}")
                logging.error(f"Image upload error: {e}")
                messagebox.showerror("Error", f"Failed to open image: {e}")

    def start_processing(self, file_path):
        """Start image processing in background thread"""
        # Reset UI
        self.result_label.config(text="Processing...")
        self.save_button.config(state=tk.DISABLED)
        self.progress.grid()
        self.progress.start()
        self.status_label.config(text="Analyzing image...")
        
        # Process in background thread
        threading.Thread(
            target=self.process_image,
            args=(file_path,),
            daemon=True
        ).start()

    def process_image(self, image_path):
        """Process image and get diagnosis"""
        try:
            # Add to recent images if not already there
            if image_path not in self.controller.recent_images:
                self.controller.recent_images.insert(0, image_path)
                if len(self.controller.recent_images) > 20:  # Limit history
                    self.controller.recent_images.pop()
            
            # Get prediction
            start_time = time.time()
            prediction = self.diagnose_disease(image_path)
            processing_time = time.time() - start_time
            
            # Update UI on main thread
            self.after(0, self.update_results, prediction, processing_time, image_path)
            
        except Exception as e:
            self.after(0, self.status_label.config, text=f"Error: {str(e)}")
            logging.error(f"Processing error: {e}")
        finally:
            self.after(0, self.progress.stop)
            self.after(0, self.progress.grid_remove)

    def update_results(self, prediction, processing_time, image_path):
        """Update UI with results"""
        self.result_label.config(text=prediction)
        self.save_button.config(state=tk.NORMAL)
        self.status_label.config(
            text=f"Analysis complete in {processing_time:.2f} seconds. Image: {os.path.basename(image_path)}"
        )
        
        # Add to diagnosis history
        self.controller.diagnosis_history.append({
            "image": image_path,
            "diagnosis": prediction,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Refresh history page
        if "HistoryPage" in self.controller.frames:
            self.controller.frames["HistoryPage"].refresh_recent_uploads()

    def display_image(self, img):
        """Display image in the preview frame"""
        try:
            # Maintain aspect ratio
            width, height = img.size
            max_size = (self.image_frame.winfo_width() - 20, 
                       self.image_frame.winfo_height() - 20)
            
            ratio = min(max_size[0]/width, max_size[1]/height)
            new_size = (int(width*ratio), int(height*ratio))
            
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk  # Keep reference
            
        except Exception as e:
            self.status_label.config(text=f"Display error: {str(e)}")
            logging.error(f"Image display error: {e}")

    def diagnose_disease(self, image_path):
        """Diagnose disease in the uploaded image"""
        if self.controller.disease_model is None:
            return "Error: Disease detection model not loaded"
        
        # First check if image is actually a crop
        if self.controller.is_non_crop_image(image_path):
            return "Error: Uploaded image doesn't appear to be a crop leaf image"
        
        try:
            # Load and preprocess the image
            img = image.load_img(image_path, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict the disease
            predictions = self.controller.disease_model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100
            
            # Get prediction details
            predicted_label = list(self.controller.class_labels.keys())[predicted_class]
            disease_name = " ".join(predicted_label.split("_")[1:])
            crop_type = predicted_label.split("_")[0]
            
            if confidence < 50:
                return (
                    f"Low confidence prediction ({confidence:.1f}%)\n"
                    f"Possible {crop_type} disease: {disease_name}\n"
                    "Please upload a clearer image for better results"
                )
            
            # Format results
            result = (
                f"Crop Type: {crop_type}\n"
                f"Disease: {disease_name}\n"
                f"Confidence: {confidence:.1f}%"
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Diagnosis error: {e}")
            return f"Error during diagnosis: {str(e)}"

    def save_report(self):
        """Save diagnosis report as PDF"""
        if not self.current_image_path:
            messagebox.showerror("Error", "No image to generate report")
            return
            
        diagnosis_text = self.result_label.cget("text")
        default_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{os.path.splitext(os.path.basename(self.current_image_path))[0]}_report.pdf"
        )
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=default_filename,
            title="Save Diagnosis Report"
        )
        
        if file_path:
            try:
                # Create PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                
                # Title
                pdf.cell(0, 10, "Crop Disease Diagnosis Report", ln=True, align="C")
                pdf.ln(10)
                
                # Report details
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                pdf.cell(0, 10, f"Image: {os.path.basename(self.current_image_path)}", ln=True)
                
                # Diagnosis
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Diagnosis Results:", ln=True)
                pdf.set_font("Arial", size=12)
                
                # Split diagnosis text into lines
                for line in diagnosis_text.split("\n"):
                    pdf.cell(0, 10, line, ln=True)
                
                # Add the image
                try:
                    # Create temporary resized image for PDF
                    img = Image.open(self.current_image_path)
                    img.thumbnail((150, 150))
                    temp_img_path = "temp_report_img.jpg"
                    img.save(temp_img_path)
                    
                    pdf.ln(10)
                    pdf.cell(0, 10, "Image Preview:", ln=True)
                    pdf.image(temp_img_path, x=50, w=100)
                    
                    # Clean up temp file
                    os.remove(temp_img_path)
                except Exception as e:
                    logging.warning(f"Couldn't embed image in PDF: {e}")
                
                # Save PDF
                pdf.output(file_path)
                messagebox.showinfo("Success", f"Report saved to:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save PDF: {e}")
                logging.error(f"PDF save error: {e}")

    # Image adjustment methods
    def adjust_brightness(self, factor):
        if self.uploaded_image:
            enhancer = ImageEnhance.Brightness(self.uploaded_image)
            self.filtered_image = enhancer.enhance(factor)
            self.display_image(self.filtered_image)

    def adjust_contrast(self, factor):
        if self.uploaded_image:
            enhancer = ImageEnhance.Contrast(self.uploaded_image)
            self.filtered_image = enhancer.enhance(factor)
            self.display_image(self.filtered_image)

    def adjust_saturation(self, factor):
        if self.uploaded_image:
            enhancer = ImageEnhance.Color(self.uploaded_image)
            self.filtered_image = enhancer.enhance(factor)
            self.display_image(self.filtered_image)

    def sharpen_image(self):
        if self.uploaded_image:
            self.filtered_image = self.uploaded_image.filter(ImageFilter.SHARPEN)
            self.display_image(self.filtered_image)

    def reset_filters(self):
        """Reset all image filters to original"""
        if self.uploaded_image:
            self.filtered_image = self.uploaded_image.copy()
            self.display_image(self.filtered_image)
            self.brightness_scale.set(1.0)
            self.contrast_scale.set(1.0)
            self.saturation_scale.set(1.0)


class HistoryPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller

        # Configure grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Title label
        title_label = ttk.Label(self, text="Recent Uploads", 
                             font=("Helvetica", 18, "bold"))
        title_label.grid(row=0, column=0, columnspan=4, pady=(20, 10), padx=10)

        # Listbox with scrollbar
        list_frame = ttk.Frame(self)
        list_frame.grid(row=1, column=0, columnspan=4, sticky="nsew", padx=10, pady=10)
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.recent_uploads = tk.Listbox(
            list_frame, 
            yscrollcommand=scrollbar.set,
            height=8,  
            font=("Helvetica", 12),
            selectbackground="#64B5F6"
        )
        scrollbar.config(command=self.recent_uploads.yview)
        
        self.recent_uploads.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Button frame
        button_frame = ttk.Frame(self)
        button_frame.grid(row=2, column=0, columnspan=4, pady=10, padx=10, sticky="ew")
        
        # Action buttons
        buttons = [
            ("View Image", "view_icon", self.view_image),
            ("Diagnose Again", "diagnose_icon", self.re_diagnose),
            ("Delete Upload", "delete_icon", self.delete_upload),
            ("Export History", "export_icon", self.export_history)
        ]
        
        for i, (text, icon_name, command) in enumerate(buttons):
            btn = ttk.Button(
                button_frame,
                text=text,
                image=controller.icon_images.get(icon_name),
                compound=tk.LEFT,
                command=command,
                style='TButton'
            )
            btn.grid(row=0, column=i, padx=5, pady=5, sticky="ew")
            btn.bind("<Enter>", lambda e, b=btn: b.config(style='Hover.TButton'))
            btn.bind("<Leave>", lambda e, b=btn: b.config(style='TButton'))
            button_frame.grid_columnconfigure(i, weight=1)

        # Back button
        back_button = ttk.Button(
            self, 
            text="Back to Home", 
            command=lambda: controller.show_frame("HomePage"), 
            style='TButton'
        )
        back_button.grid(row=3, column=0, columnspan=4, pady=10, padx=10, sticky="ew")
        back_button.bind("<Enter>", lambda e: back_button.config(style='Hover.TButton'))
        back_button.bind("<Leave>", lambda e: back_button.config(style='TButton'))

    def refresh_recent_uploads(self):
        """Update the list of recent uploads"""
        self.recent_uploads.delete(0, tk.END)
        for img in self.controller.recent_images:
            self.recent_uploads.insert(tk.END, os.path.basename(img))

    def view_image(self):
        try:
            selected_index = self.recent_uploads.curselection()[0]
            image_path = self.controller.recent_images[selected_index]
            
            top = Toplevel(self)
            top.title("Image Preview")
            
            try:
                img = Image.open(image_path)
                img.thumbnail((400, 400))  # Maintain aspect ratio
                img_tk = ImageTk.PhotoImage(img)
                
                label = tk.Label(top, image=img_tk)
                label.image = img_tk  # Keep reference
                label.pack(padx=10, pady=10)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {e}")
                top.destroy()
                
        except IndexError:
            messagebox.showerror("Error", "Please select an image first")

    def re_diagnose(self):
        try:
            selected_index = self.recent_uploads.curselection()[0]
            image_path = self.controller.recent_images[selected_index]
            
            # Switch to diagnosis page and process
            self.controller.show_frame("DiagnosisPage")
            diagnosis_page = self.controller.frames["DiagnosisPage"]
            diagnosis_page.current_image_path = image_path
            diagnosis_page.uploaded_image = Image.open(image_path)
            diagnosis_page.display_image(diagnosis_page.uploaded_image)
            
            # Start processing
            diagnosis_page.progress.grid()
            diagnosis_page.progress.start()
            threading.Thread(
                target=diagnosis_page.process_image, 
                args=(image_path,),
                daemon=True
            ).start()
            
        except IndexError:
            messagebox.showerror("Error", "Please select an image first")

    def delete_upload(self):
        try:
            selected_index = self.recent_uploads.curselection()[0]
            del self.controller.recent_images[selected_index]
            self.refresh_recent_uploads()
            
        except IndexError:
            messagebox.showerror("Error", "Please select an image to delete")

    def export_history(self):
        if not self.controller.recent_images:
            messagebox.showerror("Error", "No history to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")],
            title="Save Upload History"
        )
        
        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write("Filename,Date\n")
                    for img in self.controller.recent_images:
                        timestamp = time.strftime(
                            "%Y-%m-%d %H:%M", 
                            time.localtime(os.path.getmtime(img))
                        )
                        f.write(f"{os.path.basename(img)},{timestamp}\n")
                
                messagebox.showinfo("Success", f"History exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")

class FeedbackPage(ttk.Frame):
    def __init__(self, parent, controller):
        ttk.Frame.__init__(self, parent)
        self.controller = controller

        # Configure grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Title label
        title_label = ttk.Label(self, text="User Feedback", 
                             font=("Helvetica", 18, "bold")) 
        title_label.grid(row=0, column=0, columnspan=2, pady=(20, 10), padx=10)

        # Feedback text area with scrollbar
        text_frame = ttk.Frame(self)
        text_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        self.feedback_text = tk.Text(
            text_frame, 
            yscrollcommand=scrollbar.set,
            height=8, 
            width=50, 
            font=("Helvetica", 12), 
            padx=5, 
            pady=5,
            wrap="word"
        )
        scrollbar.config(command=self.feedback_text.yview)
        
        self.feedback_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Submit button
        submit_button = ttk.Button(
            self,
            text="Submit Feedback",
            image=controller.icon_images.get("submit_icon"),
            compound=tk.LEFT,
            command=self.submit_feedback,
            style='TButton'
        )
        submit_button.grid(row=2, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        submit_button.bind("<Enter>", lambda e: submit_button.config(style='Hover.TButton'))
        submit_button.bind("<Leave>", lambda e: submit_button.config(style='TButton'))

        # Back button
        back_button = ttk.Button(
            self, 
            text="Back to Home", 
            command=lambda: controller.show_frame("HomePage"), 
            style='TButton'
        )
        back_button.grid(row=3, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
        back_button.bind("<Enter>", lambda e: back_button.config(style='Hover.TButton'))
        back_button.bind("<Leave>", lambda e: back_button.config(style='TButton'))

    def submit_feedback(self):
        feedback = self.feedback_text.get("1.0", tk.END).strip()
        if not feedback:
            messagebox.showerror("Error", "Please enter your feedback before submitting.")
            return

        # Create feedback directory if it doesn't exist
        feedback_dir = "feedback"
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Save with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(feedback_dir, f"feedback_{timestamp}.txt")
        
        try:
            with open(file_path, "w") as f:
                f.write(feedback)
            
            messagebox.showinfo("Thank You", "Your feedback has been submitted successfully.")
            self.feedback_text.delete("1.0", tk.END)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save feedback: {e}")
            logging.error(f"Feedback save error: {e}")

class HelpPage(ttk.Frame):
    def __init__(self, parent, controller, icon_images=None):
        super().__init__(parent)
        self.controller = controller
        self.icon_images = {} 

        # Create canvas with scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor=tk.NW)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # Bind mousewheel for scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Title label
        title_label = ttk.Label(
            self.scrollable_frame, 
            text="Help & Instructions", 
            font=("Helvetica", 20, "bold"),  
        )
        title_label.pack(fill=tk.X, padx=20, pady=(20, 10))

        # Help sections
        self.help_sections = [
            {
                "title": "Getting Started",
                "icon": "help_icon",
                "content": [
                    "Welcome to the Crop Disease Diagnosis application!",
                    "This tool helps identify diseases in crop leaves using image analysis."
                ]
            },
            {
                "title": "How to Diagnose",
                "icon": "diagnose_icon",
                "content": [
                    "1. Go to the Diagnosis page from the Home screen",
                    "2. Click 'Upload Crop Image' to select an image",
                    "3. The system will analyze the image and show results",
                    "4. Use filters if needed to improve image quality",
                    "5. Save the diagnosis report when complete"
                ]
            },
            {
                "title": "Viewing History",
                "icon": "history_icon",
                "content": [
                    "• View your previously uploaded images",
                    "• Re-run diagnosis on old images",
                    "• Delete images from your history",
                    "• Export your history to a CSV file"
                ]
            },
            {
                "title": "Providing Feedback",
                "icon": "feedback_icon",
                "content": [
                    "We welcome your feedback to improve the application!",
                    "Use the Feedback page to share your thoughts."
                ]
            },
            {
                "title": "Dark Mode",
                "icon": "dark_mode_icon",
                "content": [
                    "Toggle between light and dark color schemes",
                    "Easier on the eyes in low-light conditions"
                ]
            },
            {
                "title": "Need More Help?",
                "icon": "help_icon",
                "content": [
                    "Contact support at: ayushchoubey800@gmail.com",
                    "We're happy to assist with any questions!"
                ]
            }
        ]

        # Create help sections
        for section in self.help_sections:
            self.create_section(section)

        # Back button
        back_button = ttk.Button(
            self.scrollable_frame,
            text="Back to Home",
            command=lambda: controller.show_frame("HomePage"),
            style='TButton'
        )
        back_button.pack(pady=20, ipady=5, ipadx=20)
        back_button.bind("<Enter>", lambda e: back_button.config(style='Hover.TButton'))
        back_button.bind("<Leave>", lambda e: back_button.config(style='TButton'))

    def create_section(self, section):
        """Creates a help section with icon, title and content"""
        section_frame = ttk.Frame(self.scrollable_frame)
        section_frame.pack(fill=tk.X, padx=20, pady=10)

        # Icon
        icon = self.icon_images.get(section["icon"])
        if icon:
            icon_label = ttk.Label(section_frame, image=icon)
            icon_label.image = icon  # Keep reference
            icon_label.pack(side=tk.LEFT, padx=(0, 10))

        # Content frame
        content_frame = ttk.Frame(section_frame)
        content_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Title
        title_label = ttk.Label(
            content_frame, 
            text=section["title"], 
            font=("Helvetica", 14, "bold"), 
            anchor=tk.W
        )
        title_label.pack(fill=tk.X)

        # Content items
        for item in section["content"]:
            content_label = ttk.Label(
                content_frame,
                text=item,
                font=("Helvetica", 12),
                justify="left",
                anchor=tk.W,
                wraplength=600
            )
            content_label.pack(fill=tk.X, pady=2)

        # Separator
        ttk.Separator(self.scrollable_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20, pady=10)

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except Exception as e:
        logging.error(f"Application error: {e}")
        messagebox.showerror("Fatal Error", f"The application encountered an error and will close:\n{str(e)}")
