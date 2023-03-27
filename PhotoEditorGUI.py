import globalVar
from globalVar import UNIT, HEIGHT, WIDTH, btnW, btnH
from tkinter import *
import numpy as np
from FilterLib import *
from FunctionProcess import *
import tkinter as tk

# Create a blank white image of size 500x500
globalVar.img = np.zeros((500, 500, 3), dtype=np.uint8)
globalVar.img.fill(255)
globalVar.img_original = globalVar.img.copy()
globalVar.img_edited = globalVar.img.copy()

window = Tk()
window.title("Photo editor")
x = (window.winfo_screenwidth() // 2) - (WIDTH // 2)
y = (window.winfo_screenheight() // 2) - (HEIGHT // 2)
window.geometry(f"{WIDTH}x{HEIGHT}+{x}+{y}")
window.minsize(WIDTH, HEIGHT)
window.maxsize(WIDTH, HEIGHT)

#=========FUNCTION===============
# Create a new button to reset the image
def reset_image():
    globalVar.img = globalVar.img_original.copy()
    load_img_to_frame(globalVar.img, originalFrame)
    globalVar.img_edited = globalVar.img_original.copy()
    load_img_to_frame(globalVar.img_edited, editedFrame)
    scale_logTransformations.set(1)
    scale_gammaTransformationsY.set(1)
    scale_gammaTransformationsC.set(1)
    scale_dilation_filter.set(0)
    scale_erosion_filter.set(0)
    scale_ksize_dilation_erosion.set(0)
    
def apply_image():
    globalVar.img = globalVar.img_edited.copy()
    load_img_to_frame(globalVar.img, originalFrame)

def undo_image():
    globalVar.img_edited = globalVar.img.copy()
    load_img_to_frame(globalVar.img_edited, editedFrame)

def print_mouse_position(event):
    x, y = event.y, event.x
    print(f"Mouse position: x={x}, y={y}")
    return x, y   

def changeToFillMode():
    globalVar.isFill = not globalVar.isFill
    print(globalVar.isFill)
    if globalVar.isFill:
        button_regionFilling.config(bg="#24FC69")
    else:
        button_regionFilling.config(bg="#f0f0f0")

def print_default():
    button_apply.place(x = WIDTH / 4 + 0*UNIT, y = 0*UNIT)
    button_undo.place(x = WIDTH / 4 + 3*UNIT, y = 0*UNIT)
    button_reset.place(x = WIDTH / 4 + 6*UNIT, y = 0*UNIT)

def clear_all_widget(frame):
    for widget in frame.winfo_children():
            widget.place_forget()
    print_default()
    
def open_spatialDomain():
    clear_all_widget(optionFrame)
    pad_left = WIDTH / 6
    button_gray.place(x = 0.5*UNIT + pad_left, y = 1*UNIT)
    button_negative.place(x = 0.5*UNIT + pad_left, y = 1.5*UNIT)
    button_blur.place(x = 0.5*UNIT + pad_left, y = 2*UNIT)
    button_Gaussian_blur.place(x = 0.5*UNIT + pad_left, y = 2.5*UNIT)
    button_logTransformations.place(x = 0.5*UNIT + pad_left, y = 3*UNIT)
    button_gammaTransformations.place(x = 0.5*UNIT + pad_left, y = 3.5*UNIT)
    
    button_median_filter.place(x = 4.5*UNIT + pad_left, y = 1*UNIT)
    button_max_filter.place(x = 4.5*UNIT + pad_left, y = 1.5*UNIT)
    button_min_filter.place(x = 4.5*UNIT + pad_left, y = 2*UNIT)
    button_midpoint_filter.place(x = 4.5*UNIT + pad_left, y = 2.5*UNIT)
    button_laplacian_filter.place(x = 4.5*UNIT + pad_left, y = 3*UNIT)
    button_sobel_filter.place(x = 4.5*UNIT + pad_left, y = 3.5*UNIT)
    button_sobel_edge_filter.place(x = 4.5*UNIT + pad_left, y = 4*UNIT)
    
    scale_logTransformations.place(x=8.5*UNIT + pad_left, y=1*UNIT)
    scale_gammaTransformationsY.place(x=8.5*UNIT + pad_left, y=2.25*UNIT)
    scale_gammaTransformationsC.place(x=8.5*UNIT + pad_left, y=3.5*UNIT)

def open_frequencyDomain():
    clear_all_widget(optionFrame)
    pad_left = WIDTH / 4 + 3*UNIT
    button_freq_lowpass_filters.place(x = 0*UNIT + pad_left, y = 1*UNIT)
    button_freq_butterworth_lp_filters.place(x = 0*UNIT + pad_left, y = 1.5*UNIT)
    button_freq_gaussian_lp_filters.place(x = 0*UNIT + pad_left, y = 2*UNIT)
    button_freq_highpass_filters.place(x = 0*UNIT + pad_left, y = 2.5*UNIT)
    button_freq_butterworth_hp_filters.place(x = 0*UNIT + pad_left, y = 3*UNIT)
    button_freq_gaussian_hp_filters.place(x = 0*UNIT + pad_left, y = 3.5*UNIT)

def open_morphologicalProcess():
    clear_all_widget(optionFrame)
    pad_left = WIDTH / 6
    button_erosion_filter.place(x = 0.5*UNIT + pad_left, y = 1*UNIT)
    button_dilation_filter.place(x = 0.5*UNIT + pad_left, y = 1.5*UNIT)
    button_getBoundaryExtraction.place(x = 0.5*UNIT + pad_left, y = 2*UNIT)
    button_regionFilling.place(x = 0.5*UNIT + pad_left, y = 2.5*UNIT)
    
    scale_erosion_filter.place(x=8.5*UNIT + pad_left, y=1*UNIT)
    scale_dilation_filter.place(x=8.5*UNIT + pad_left, y=2.25*UNIT)
    scale_ksize_dilation_erosion.place(x=8.5*UNIT + pad_left, y=3.5*UNIT)

#=========FUNCTION===============

#=========CREATE WIDGET==========
originalFrame = Frame(window, bg='white', width=10*UNIT, height=7*UNIT)
originalFrame.place(x=0.25*UNIT, y=0)

editedFrame = Frame(window, bg='white', width=10*UNIT, height=7*UNIT)
editedFrame.place(x=10.75*UNIT, y=0)

optionFrame = Frame(window, bg='cyan', width=WIDTH, height=5*UNIT)
optionFrame.place(x=0, y=7.25*UNIT)

# Create a new button in optionFrame to open an image in originalFrame
button_open = Button(optionFrame, text='Open Image', command=lambda: open_image(originalFrame, editedFrame), width=btnW, height=btnH)
# Create a new button to convert image to grayscale
button_gray = Button(optionFrame, text='Convert to Gray', command=lambda: convert_to_Gray(editedFrame), width=btnW, height=btnH)
# Create a new button to negative the image
button_negative = Button(optionFrame, text='Negative Image', command=lambda: negative_image(editedFrame), width=btnW, height=btnH)
scale_logTransformations = Scale(optionFrame, orient=HORIZONTAL, from_= 0, to=45.9, resolution=0.1,
    length=4*UNIT, label="C in Log Transformations"
)
scale_logTransformations.set(1)
# Create a button to log transformations
button_logTransformations = Button(optionFrame, text='Log Transformations', command=lambda: logTransformations(editedFrame, scale_logTransformations.get()), width=btnW, height=btnH)
# Create a scale to y in gamma transformations
scale_gammaTransformationsY = Scale(optionFrame, orient=HORIZONTAL, from_= 0.01, to=3, resolution=0.001,
    length=4*UNIT, label="Y in Gamma Transformations"
)
scale_gammaTransformationsY.set(1)
# Create a scale to c in gamma transformations
scale_gammaTransformationsC = Scale(optionFrame, orient=HORIZONTAL, from_= 0.01, to=5, resolution=0.001,
    length=4*UNIT, label="C in Gamma Transformations"
)
scale_gammaTransformationsC.set(1)
# Create a button to gamma transformations
button_gammaTransformations = Button(optionFrame, text='Gamma Transformations', command=lambda: gammaTransformations(editedFrame, scale_gammaTransformationsY.get(), scale_gammaTransformationsC.get()), width=btnW, height=btnH)
# Create a new button to save the image
button_save = Button(optionFrame, text='Save Image', command=save_image, width=btnW, height=btnH, bg="cyan")
# Create a new button to blur the image
button_blur = Button(optionFrame, text='Blur Box Image', command=lambda: blur_image(editedFrame), width=btnW, height=btnH)
# Create a new button to Gaussian blur the image
button_Gaussian_blur = Button(optionFrame, text='Blur Gaussian Image', command=lambda: blur_Gaussian_image(editedFrame), width=btnW, height=btnH)
# Create a new button to median filter the image
button_median_filter = Button(optionFrame, text='Median filter', command=lambda: median_filter_image(editedFrame), width=btnW, height=btnH)
# Create a new button to max filter the image
button_max_filter = Button(optionFrame, text='Max filter', command=lambda: max_filter_image(editedFrame), width=btnW, height=btnH)
# Create a new button to min filter the image
button_min_filter = Button(optionFrame, text='Min filter', command=lambda: min_filter_image(editedFrame), width=btnW, height=btnH)
# Create a new button to midpoint filter the image
button_midpoint_filter = Button(optionFrame, text='Midpoint filter', command=lambda: midpoint_filter_image(editedFrame), width=btnW, height=btnH)
# Create a new button to Laplacian filter
button_laplacian_filter = Button(optionFrame, text='Laplacian filter', command=lambda: laplacian_filter_image(editedFrame), width=btnW, height=btnH)
# Create a new button to sobel filter
button_sobel_filter = Button(optionFrame, text='Sobel filter', command=lambda: sobel_filter_image(editedFrame), width=btnW, height=btnH)
# Create a new button to sobel edge filter
button_sobel_edge_filter = Button(optionFrame, text='Sobel edge filter', command=lambda: sobel_edge_filter_image(editedFrame), width=btnW, height=btnH)
# Create a new button to frequency Lowpass Filters
button_freq_lowpass_filters = Button(optionFrame, text='Freq Lowpass filter', command=lambda: freq_lowpass_filters(editedFrame), width=btnW, height=btnH)
# Create a new button to frequency Butterworth Lowpass Filters
button_freq_butterworth_lp_filters = Button(optionFrame, text='Freq Butterworth Lp', command=lambda: freq_butterworth_lp_filters(editedFrame), width=btnW, height=btnH)
# Create a new button to frequency Gaussian Lowpass Filters
button_freq_gaussian_lp_filters = Button(optionFrame, text='Freq Gaussian Lp', command=lambda: freq_gaussian_lp_filters(editedFrame), width=btnW, height=btnH)
# Create a new button to frequency Highpass Filters
button_freq_highpass_filters = Button(optionFrame, text='Freq Highpass filter', command=lambda: freq_highpass_filters(editedFrame), width=btnW, height=btnH)
# Create a new button to frequency Butterworth Highpass Filters
button_freq_butterworth_hp_filters = Button(optionFrame, text='Freq Butterworth Hp', command=lambda: freq_butterworth_hp_filters(editedFrame), width=btnW, height=btnH)
# Create a new button to frequency Gaussian Highpass Filters
button_freq_gaussian_hp_filters = Button(optionFrame, text='Freq Gaussian Hp', command=lambda: freq_gaussian_hp_filters(editedFrame), width=btnW, height=btnH)
# Create a scale to Erosion filter
scale_erosion_filter = Scale(optionFrame, orient=HORIZONTAL, from_= 0, to=255, resolution=1,
    length=4*UNIT, label="Threshold in Erosion filter"
)
scale_erosion_filter.set(0)
# Create a button to Erosion filter
button_erosion_filter = Button(optionFrame, text='Erosion filter', command=lambda: erosion_filter(editedFrame, scale_erosion_filter.get(), scale_ksize_dilation_erosion.get()), width=btnW, height=btnH)
# Create a scale to Dilation filter
scale_dilation_filter = Scale(optionFrame, orient=HORIZONTAL, from_= 0, to=255, resolution=1,
    length=4*UNIT, label="Threshold in Dilation filter"
)
scale_dilation_filter.set(0)
# Create a button to Dilation filter
button_dilation_filter = Button(optionFrame, text='Dilation filter', command=lambda: dilation_filter(editedFrame, scale_erosion_filter.get(), scale_ksize_dilation_erosion.get()), width=btnW, height=btnH)
# Create a scale to Dilation and Erosion filter
scale_ksize_dilation_erosion = Scale(optionFrame, orient=HORIZONTAL, from_= 0, to=255, resolution=1,
    length=4*UNIT, label="Kernel size for Dilation and Erosion"
)
scale_ksize_dilation_erosion.set(0)

# Create a button to  get Boundary Extraction
button_getBoundaryExtraction = Button(optionFrame, text='Boundary Extraction', command=lambda: get_BoundaryExtraction(editedFrame), width=btnW, height=btnH)

# Create a button to region filling
button_regionFilling = Button(optionFrame, text='Fill', command=changeToFillMode, width=btnW, height=btnH)


# Create a new button to apply the image
button_apply = Button(optionFrame, text='Apply', command=apply_image, width=btnW, height=btnH, bg="#14F786")
# Create a new button to undo the image
button_undo = Button(optionFrame, text='Undo', command=undo_image, width=btnW, height=btnH, bg="#09B9F6")
# Create a new button to reset the image
button_reset = Button(optionFrame, text='Reset Image', command=reset_image, width=btnW, height=btnH, bg="#F71A3F")
# editedFrame.bind("<Button-1>", print_mouse_position)

#=========CREATE WIDGET AND ADD FUNCTION==========
load_img_to_frame(globalVar.img, originalFrame)
load_img_to_frame(globalVar.img_edited, editedFrame)
print_default()
# create a menubar
menubar = Menu(window)
filemenu = Menu(menubar, tearoff=0, font = globalVar.FRONT)
filemenu.add_command(label="Open", command= lambda: open_image(originalFrame, editedFrame))
filemenu.add_command(label="Save", command=save_image)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=window.quit)
menubar.add_cascade(label="File", menu=filemenu)

# Spatial Domain
menubar.add_command(label="Spatial Domain", command= open_spatialDomain)

# Frequency Domain
menubar.add_command(label="Frequency Domain", command= open_frequencyDomain)

# Morphological Process
menubar.add_command(label="Morphological Process", command= open_morphologicalProcess)

# add Apply, Undo, Save, and Reset items to menubar
menubar.add_command(label="Apply", command=apply_image)
menubar.add_command(label="Undo", command=undo_image)
menubar.add_command(label="Reset", command=reset_image)
menubar.add_command(label="Save", command=save_image)

window.config(menu=menubar, background='cyan')
window.mainloop()
