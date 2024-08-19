import dlib
import cv2
import numpy as np
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk


# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

class Face_Register:
    def __init__(self):
        self.current_frame_faces_cnt = 0  # Count of faces in current frame
        self.existing_faces_cnt = 0  # Count of saved faces
        self.ss_cnt = 0  # Count for screen shots

        # Tkinter GUI setup
        self.win = tk.Tk()
        self.win.title("Student Register")
        self.win.geometry("1000x500")

        # GUI left part (camera feed)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)

        # GUI right part (controls and info)
        self.frame_right_info = tk.Frame(self.win)
        self.frame_right_info.pack(side=tk.RIGHT)

        # Labels for displaying information
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text="0")
        self.label_fps_info = tk.Label(self.frame_right_info, text="")
        self.label_face_cnt = tk.Label(self.frame_right_info, text="0")
        self.label_warning = tk.Label(self.frame_right_info)
        self.log_all = tk.Label(self.frame_right_info)

        # Input field for name
        self.input_name = tk.Entry(self.frame_right_info)
        self.input_name_char = ""

        # Fonts
        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        # Paths and directories
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""

        # Face ROI (Region of Interest) information
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        # Flags
        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        # FPS calculation
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # IP camera settings
        self.ip_camera_url = "rtsp://192.168.186.244:554/stream1"  # Update this URL with your camera's address
        self.username = "admin123"  # Update with your camera's username
        self.password = "password123"  # Update with your camera's password
        self.cap = None  # Will be initialized in get_frame method
        
        
        
    
        

    def GUI_info(self):
        # Set up the GUI elements
        tk.Label(self.frame_right_info, text="Face Register", font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=20)
        
        tk.Label(self.frame_right_info, text="FPS: ").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(self.frame_right_info, text="Faces in database: ").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(self.frame_right_info, text="Faces in current frame: ").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(self.frame_right_info, text="Step 1: Clear face photos", font=self.font_step_title).grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        tk.Button(self.frame_right_info, text='Clear', command=self.GUI_clear_data).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        
        tk.Label(self.frame_right_info, text="Step 2: Input name", font=self.font_step_title).grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        tk.Label(self.frame_right_info, text="Name: ").grid(row=8, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_name.grid(row=8, column=1, sticky=tk.W, padx=0, pady=2)
        tk.Button(self.frame_right_info, text='Input', command=self.GUI_get_input_name).grid(row=8, column=2, padx=5)
        
        tk.Label(self.frame_right_info, text="Step 3: Save face image", font=self.font_step_title).grid(row=9, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        tk.Button(self.frame_right_info, text='Save current face', command=self.save_current_face).grid(row=10, column=0, columnspan=3, sticky=tk.W)
        
        self.log_all.grid(row=11, column=0, columnspan=20, sticky=tk.W, padx=5, pady=20)

    def GUI_clear_data(self):
        # Clear all saved face data
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        self.label_cnt_face_in_database['text'] = "0"
        self.existing_faces_cnt = 0
        self.log_all["text"] = "Face images and features_all.csv removed!"

    def GUI_get_input_name(self):
        # Get the input name and create a new face folder
        self.input_name_char = self.input_name.get()
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)

    def pre_work_mkdir(self):
        # Create the main directory for storing face data if it doesn't exist
        if not os.path.isdir(self.path_photos_from_camera):
            os.mkdir(self.path_photos_from_camera)

    def check_existing_faces_cnt(self):
        person_num_list = []
        for person_order in os.listdir(self.path_photos_from_camera):
            parts = person_order.split('_')
            if len(parts) >= 2 and parts[0] == "person" and parts[1].isdigit():
                person_num_list.append(int(parts[1]))
        self.existing_faces_cnt = max(person_num_list) if person_num_list else 0

    def create_face_folder(self):
        # Create a new folder for the current face
        self.existing_faces_cnt += 1
        if self.input_name_char:
            self.current_face_dir = f"{self.path_photos_from_camera}person_{self.existing_faces_cnt}_{self.input_name_char}"
        else:
            self.current_face_dir = f"{self.path_photos_from_camera}person_{self.existing_faces_cnt}"
        os.makedirs(self.current_face_dir)
        self.log_all["text"] = f"\"{self.current_face_dir}/\" created!"
        logging.info("%-40s %s", "Create folders:", self.current_face_dir)
        self.ss_cnt = 0
        self.face_folder_created_flag = True

    def save_current_face(self):
        # Save the current face ROI as an image
        if self.face_folder_created_flag:
            if self.current_frame_faces_cnt == 1:
                if not self.out_of_range_flag:
                    self.ss_cnt += 1
                    self.face_ROI_image = self.current_frame[
                        self.face_ROI_height_start-self.hh:self.face_ROI_height_start+self.face_ROI_height+self.hh,
                        self.face_ROI_width_start-self.ww:self.face_ROI_width_start+self.face_ROI_width+self.ww
                    ]
                    self.face_ROI_image = cv2.resize(self.face_ROI_image, (self.face_ROI_width * 2, self.face_ROI_height * 2))
                    # Convert from BGR to RGB
                    self.face_ROI_image = cv2.cvtColor(self.face_ROI_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(f"{self.current_face_dir}/img_face_{self.ss_cnt}.jpg", self.face_ROI_image)
                    self.log_all["text"] = f"\"{self.current_face_dir}/img_face_{self.ss_cnt}.jpg\" saved!"
                    logging.info("%-40s %s/img_face_%s.jpg", "Save into：", self.current_face_dir, self.ss_cnt)
                else:
                    self.log_all["text"] = "Please do not out of range!"
            else:
                self.log_all["text"] = "No face in current frame!"
        else:
            self.log_all["text"] = "Please run step 2!"

    def get_frame(self):
        # Initialize video capture if not already done
        if self.cap is None:
            # Construct the RTSP URL with authentication
            auth_url = f"rtsp://{self.username}:{self.password}@{self.ip_camera_url.split('//')[1]}"
            self.cap = cv2.VideoCapture(auth_url)
        
        # Read a frame from the video stream
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            return False, None

    def update_fps(self):
        # Calculate and update FPS
        now = time.time()
        # Update FPS count every second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

        self.label_fps_info["text"] = str(round(self.fps_show, 2))

    def process(self):
        # Main processing loop
        ret, self.current_frame = self.get_frame()
        if ret:
            faces = detector(self.current_frame, 0)
            self.current_frame_faces_cnt = len(faces)

            self.update_fps()
            self.label_face_cnt["text"] = str(self.current_frame_faces_cnt)

            if self.current_frame_faces_cnt != 0:
                for k, d in enumerate(faces):
                    self.face_ROI_width_start = d.left()
                    self.face_ROI_height_start = d.top()
                    self.face_ROI_height = (d.bottom() - d.top())
                    self.face_ROI_width = (d.right() - d.left())
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)

                    # Check if face is out of frame
                    if (d.right() + self.ww) > 640 or (d.bottom() + self.hh > 480) or (d.left() - self.ww < 0) or (d.top() - self.hh < 0):
                        self.label_warning["text"] = "OUT OF RANGE"
                        self.label_warning['fg'] = 'red'
                        self.out_of_range_flag = True
                        color_rectangle = (255, 0, 0)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = ""
                        color_rectangle = (255, 255, 255)
                    
                    # Draw rectangle around face
                    cv2.rectangle(self.current_frame,
                                  (d.left() - self.ww, d.top() - self.hh),
                                  (d.right() + self.ww, d.bottom() + self.hh),
                                  color_rectangle, 2)

            # Convert frame to PhotoImage for display
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)

        # Schedule the next frame processing
        self.win.after(20, self.process)

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.GUI_info()
        self.process()
        self.win.mainloop()

def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()

if __name__ == '__main__':
    main()