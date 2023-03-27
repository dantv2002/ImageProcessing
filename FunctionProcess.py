import globalVar
from globalVar import UNIT
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2
from FilterLib import *

def print_mouse_position(event):
    x, y = event.y, event.x
    print(f"Mouse position: x={x}, y={y}")
    return x, y   

#Load to edited frame
def load_img_to_frame(img, frame, width = 10, height = 7):
    # Clear the existing image in editedFrame
    for widget in frame.winfo_children():
        widget.destroy()
    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_image = pil_image.resize((width * UNIT, height * UNIT))
    # Convert the PIL image to tkinter format
    img_edited_local = ImageTk.PhotoImage(pil_image)
    img_edited_label = Label(frame, image=img_edited_local)
    img_edited_label.image = img_edited_local
    img_edited_label.place(x = 0, y = 0)
    globalVar.childEditedFrame = img_edited_label
    globalVar.childEditedFrame.bind("<Button-1>", lambda event: fill_by_mouse_position(event, frame))
    
# Create a new button in optionFrame to open an image in originalFrame
def open_image(originalFrame, editedFrame):
    # Open file dialog to select image
    
    globalVar.file_path = filedialog.askopenfilename()
    # Load image into originalFrame
    if globalVar.file_path:
        # Clear the existing image, if any
        for widget in originalFrame.winfo_children():
            widget.destroy()

        # Load the new image
        globalVar.img = cv2.imread(globalVar.file_path)
        globalVar.original_height, globalVar.original_width, _ = globalVar.img.shape
        globalVar.scale_factor_width = globalVar.original_width / (10 * UNIT)
        globalVar.scale_factor_height = globalVar.original_height / (7 * UNIT)
        print( globalVar.scale_factor_height)
        print( globalVar.scale_factor_width )
        load_img_to_frame(globalVar.img, originalFrame)
        #set img_origin
        globalVar.img_original = globalVar.img
        # Load to edited image
        globalVar.img_edited = globalVar.img
        load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to convert image to grayscale
def convert_to_Gray(editedFrame):
    # Convert the image to grayscale
    globalVar.img_edited = convert_to_gray(globalVar.img_edited)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to negative the image
def negative_image(editedFrame):
    globalVar.img_edited = negative(globalVar.img_edited)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a scale to log transformations
def logTransformations(editedFrame, c_log):
    globalVar.img_edited = logTransformation(globalVar.img_edited, c_log)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a scale to y in gamma transformations
def gammaTransformations(editedFrame, gamma, c):
    globalVar.img_edited = gammaTransformation(globalVar.img_edited, gamma, c)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to save the image
def save_image():
    img_save = Image.fromarray(cv2.cvtColor(globalVar.img, cv2.COLOR_BGR2RGB))
    file_name = filedialog.asksaveasfilename(defaultextension=".jpg",
                                             filetypes=[("JPEG", ".jpg"), ("PNG", ".png"), ("BMP", ".bmp")])
    if not file_name:
        return
    
    # Save image in the specified file format
    if file_name:
        img_save.save(file_name)

# Create a new button to blur the image
def blur_image(editedFrame):
    globalVar.img_edited = blur_box(globalVar.img_edited)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to Gaussian blur the image
def blur_Gaussian_image(editedFrame):
    globalVar.img_edited = blur_Gaussian(globalVar.img_edited)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to median filter the image
def median_filter_image(editedFrame):
    globalVar.img_edited = median_filter(globalVar.img_edited)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to max filter the image
def max_filter_image(editedFrame):
    globalVar.img_edited = max_filter(globalVar.img_edited)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to min filter the image
def min_filter_image(editedFrame):
    globalVar.img_edited = min_filter(globalVar.img_edited)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to midpoint filter the image
def midpoint_filter_image(editedFrame):
    globalVar.img_edited = midpoint_filter(globalVar.img_edited)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to Laplacian filter
def laplacian_filter_image(editedFrame):
    globalVar.img_edited = laplacian_filter(globalVar.img_edited)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to sobel filter
def sobel_filter_image(editedFrame):
    globalVar.img_edited = sobel_filter(globalVar.img_edited)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to sobel edge filter
def sobel_edge_filter_image(editedFrame):
    globalVar.img_edited = sobel_edge_candidate_filter(globalVar.img_edited, 100)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to frequency Lowpass Filters
def freq_lowpass_filters(editedFrame):
    globalVar.img_edited = freq_filters(globalVar.img_edited, 0)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to frequency Butterworth Lowpass Filters
def freq_butterworth_lp_filters(editedFrame):
    globalVar.img_edited = freq_filters(globalVar.img_edited, 1)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to frequency Gaussian Lowpass Filters
def freq_gaussian_lp_filters(editedFrame):
    globalVar.img_edited = freq_filters(globalVar.img_edited, 2)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to frequency Highpass Filters
def freq_highpass_filters(editedFrame):
    globalVar.img_edited = freq_filters(globalVar.img_edited, 3)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to frequency Butterworth Highpass Filters
def freq_butterworth_hp_filters(editedFrame):
    globalVar.img_edited = freq_filters(globalVar.img_edited, 4)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a new button to frequency Gaussian Highpass Filters
def freq_gaussian_hp_filters(editedFrame):
    globalVar.img_edited = freq_filters(globalVar.img_edited, 5)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a scale to Erosion filter
def erosion_filter(editedFrame, threshold, ksize):
    globalVar.img_edited = erosion(globalVar.img_edited, threshold, ksize)
    load_img_to_frame(globalVar.img_edited, editedFrame)

# Create a scale to Dilation filter
def dilation_filter(editedFrame, threshold, ksize):
    globalVar.img_edited = dilation(globalVar.img_edited, threshold, ksize)
    load_img_to_frame(globalVar.img_edited, editedFrame)
    
def get_BoundaryExtraction(editedFrame):
    globalVar.img_edited = boundaryExtraction(globalVar.img_edited)
    load_img_to_frame(globalVar.img_edited, editedFrame)

def fill_by_mouse_position(event, frame):
    x, y = print_mouse_position(event)
    onClickRegionFilling(frame, x, y)
    
def onClickRegionFilling(editedFrame, x, y):
    if not globalVar.isFill:
        return
    y = (int) (y * globalVar.scale_factor_width)
    x = (int) (x * globalVar.scale_factor_height)
    print("Original x: ", x * globalVar.scale_factor_width , " y: ", y * globalVar.scale_factor_height)
    globalVar.img_edited = regionFilling(globalVar.img_edited, x, y)
    load_img_to_frame(globalVar.img_edited, editedFrame)