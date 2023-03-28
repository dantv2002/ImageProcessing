import numpy as np
import cv2
from scipy.ndimage import convolve

def convolution(img, kernel, padding = 0):
    kh,kw=kernel.shape
    if padding>0:
        h,w=img.shape
        B=np.ones((h+kh-1,w+kw-1))
        th=int(kh/2)
        tw=int(kw/2)
        B[th:h+th,tw:w+tw]=img
        img=B
    h,w=img.shape
    C=np.ones((h,w))
    for i in range(0,h-kh+1):
        for j in range(0,w-kw+1):
            sA=img[i:i+kh,j:j+kw]
            C[i,j]=np.sum(kernel*sA)
    C=C[0:h-kh+1,0:w-kw+1]
    return C

def convert_to_gray(img):
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    return img_gray

def negative(img):
    img_neg = 256 - 1 - img
    return img_neg

def blur_box(img):
    k = np.ones((5, 5))/25
    r,g,b = cv2.split(img)
    R=convolve(r,k)
    G=convolve(g,k)
    B=convolve(b,k)
    img_blur=cv2.merge((R,G,B))
    img_blur = np.array(img_blur, dtype = 'uint8')
    return img_blur

def Gausskernel(l=5, sig=1.5):
    s=round((l - 1)/2)
    ax = np.linspace(-s, s, l)
    gauss = np.exp(-np.square(ax) / (2*(sig**2)))
    kernel = np.outer(gauss, gauss)
    #tính tích the outer product of two vectors.
    return kernel / np.sum(kernel)

def blur_Gaussian(img):
    k = Gausskernel()
    r,g,b = cv2.split(img)
    R=convolve(r,k)
    G=convolve(g,k)
    B=convolve(b,k)
    img_blur=cv2.merge((R,G,B))
    img_blur = np.array(img_blur, dtype = 'uint8')
    return img_blur

def logTransformation(img, c):
    img_edited_arr = np.array(img, 'float')
    # the transformed image
    log_img = c*np.log(img_edited_arr+1)
    log_img = np.array(log_img, dtype='uint8')
    return log_img

def gammaTransformation(img, gamma, c):
    img_edited_arr = np.array(img, 'float')
    # the transformed image
    gamma_img = c*(img_edited_arr**gamma)
    gamma_img = np.array(gamma_img, dtype='uint8')
    return gamma_img

def median_filter_gray(img, kernel):
    kh,kw=kernel.shape
    h,w=img.shape
    C=np.ones((h,w))
    for i in range(0,h-kh+1):
        for j in range(0,w-kw+1):
            sA=img[i:i+kh,j:j+kw]
            C[i,j]=np.median(sA)
    return C

def median_filter(img):
    kernel = np.ones((5, 5))/25
    (r,g,b) = cv2.split(img)
    R=median_filter_gray(r, kernel)
    G=median_filter_gray(g, kernel)
    B=median_filter_gray(b, kernel)
    C = cv2.merge((R,G,B))
    C = np.array(C, dtype = 'uint8')
    return C

def max_filter_gray(img, kernel):
    kh,kw=kernel.shape
    h,w=img.shape
    C=np.ones((h,w))
    for i in range(0,h-kh+1):
        for j in range(0,w-kw+1):
            sA=img[i:i+kh,j:j+kw]
            C[i,j]=np.max(sA)
    return C

def max_filter(img):
    kernel = np.ones((5, 5))/25
    (r,g,b) = cv2.split(img)
    R=max_filter_gray(r, kernel)
    G=max_filter_gray(g, kernel)
    B=max_filter_gray(b, kernel)
    C = cv2.merge((R,G,B))
    C = np.array(C, dtype = 'uint8')
    return C

def min_filter_gray(img, kernel):
    kh,kw=kernel.shape
    h,w=img.shape
    C=np.ones((h,w))
    for i in range(0,h-kh+1):
        for j in range(0,w-kw+1):
            sA=img[i:i+kh,j:j+kw]
            C[i,j]=np.min(sA)
    return C

def min_filter(img):
    kernel = np.ones((5, 5))/25
    (r,g,b) = cv2.split(img)
    R=min_filter_gray(r, kernel)
    G=min_filter_gray(g, kernel)
    B=min_filter_gray(b, kernel)
    C = cv2.merge((R,G,B))
    C = np.array(C, dtype = 'uint8')
    return C

def min_max_filter(img, kernel):
    kh, kw = kernel.shape
    h, w, c = img.shape
    C = np.ones((h, w, c))
    for i in range(0, h-kh+1):
        for j in range(0, w-kw+1):
            sA = img[i:i+kh, j:j+kw, :]
            C[i, j, :] = [np.min(sA[:,:,c]), np.max(sA[:,:,c]), np.mean(sA[:,:,c])]
    return C

def minmax_filter(img):
    kernel = np.ones((5, 5))/25
    C = min_max_filter(img, kernel)
    return C

def midpoint_filter(img, kernel):
    kh, kw = kernel.shape
    h, w, c = img.shape
    C = np.zeros((h, w, c), dtype=np.float64)
    pad_h, pad_w = kh//2, kw//2
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    for i in range(pad_h, h+pad_h):
        for j in range(pad_w, w+pad_w):
            sA = img_padded[i-pad_h:i+pad_h+1, j-pad_w:j+pad_w+1, :]
            min_val = np.min(sA, axis=(0,1))
            max_val = np.max(sA, axis=(0,1))
            midpoint = (min_val + max_val) / 2
            C[i-pad_h, j-pad_w, :] = midpoint
    return np.uint8(C)

def midpoint_filter(img):
    # kernel = np.ones((5, 5))/25
    # (r,g,b) = cv2.split(img)
    # R=midpoint_filter(r, kernel)
    # G=midpoint_filter(g, kernel)
    # B=midpoint_filter(b, kernel)
    # C = cv2.merge((R,G,B))
    # C = np.array(C, dtype = 'uint8')
    kernel_size = 3
    C = cv2.medianBlur(img, kernel_size)
    return C

def laplacian_filter(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    # img=resizeImage(img, 0.3)
    ksize=(11,11)
    img = cv2.blur(img, ksize, cv2.BORDER_DEFAULT)
    # Tạo các kernal k1 4 hướng và k2 8 hướng
    k1=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    k2=np.array([[1,1,1],[1,-8,1],[1,1,1]])
    L1=convolve(img,k1)
    L2=convolve(img,-k1)+img
    L3=convolve(img,k2)+img
    L3 = cv2.cvtColor(np.array(L3), cv2.COLOR_GRAY2RGB)
    return L3

def Conv(img,k):
    Input=np.array(img,dtype='single')
    if Input.ndim >2:
        Out=np.zeros_like(Input)
        for i in range(3):
            Out[:,:,i]=np.convolve(Input[:,:,i],k)
    else:
        Out=convolve(Input,k)
    return Out

def sobel_filter(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = np.array(img,dtype='single')
    ky=np.array([[-1.0,-2,-1],[0,0,0],[1,2,1]])
    kx=np.transpose(ky)
    Gx=convolve(img,kx)
    Gy=convolve(img,ky)
    # print(Gy)
    Gm=np.sqrt(Gx**2+Gy**2)
    # Gm=np.array(Gm, dtype=np.uint8)
    # Out=Gm>200 # Thiết lập ngưỡng đơn lọc edge candidate
    # # plt.imshow(Out,cmap='gray'); plt.axis('off')
    # # plt.show()
    # Out = Out.astype(int) * 254
    Out = cv2.cvtColor(np.array(Gm, dtype=np.uint8),cv2.COLOR_GRAY2RGB)
    # Gm = cv2.cvtColor(Gm, cv2.COLOR_BGR2RGB)
    return Out

def sobel_edge_candidate_filter(img, threshold):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = np.array(img,dtype='single')
    ky=np.array([[-1.0,-2,-1],[0,0,0],[1,2,1]])
    kx=np.transpose(ky)
    Gx=convolve(img,kx)
    Gy=convolve(img,ky)
    Gm=np.sqrt(Gx**2+Gy**2)
    Out=Gm>threshold # Thiết lập ngưỡng đơn lọc edge candidate
    Out = Out.astype(int) * 254
    Out = cv2.cvtColor(np.array(Out, dtype=np.uint8),cv2.COLOR_GRAY2RGB)
    return Out
    
def freq_filters(img, mode = 0):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    F = np.fft.fft2(img) # convert
    F = np.fft.fftshift(F)
    M, N = img.shape
    D0 = 30
    n = 2

    u = np.arange(0, M) - M/2
    v = np.arange(0, N) - N/2
    [V, U] = np.meshgrid(v, u)
    D = np.sqrt(np.power(U, 2) + np.power(V, 2))
    H = np.array(D<=D0, 'float') # frequency Lowpass Filters
    if(mode == 1):
        H = 1/np.power(1+(D/(D0 + 0.001)), (2*n)) #frequency Butterworth Lowpass Filter
    elif(mode == 2):
        H = np.exp(-D**2/(2*D0**2)) #frequency Gaussian Lowpass Filter
    elif(mode == 3):
        H = np.array(D>=D0, 'float') # frequency Highpass Filters
    elif(mode == 4):
        H = 1/np.power(1+(D0/(D + 0.001)), (2*n)) #frequency Butterworth Highpass Filter
    elif(mode == 5):
        H = 1 - np.exp(-D**2/(2*D0**2)) # frequency Highpass Filters
    G = H*F
    G = np.fft.ifftshift(G)
    imgOut = np.real(np.fft.ifft2(G))
    imgOut = cv2.cvtColor(np.array(imgOut, dtype=np.uint8),cv2.COLOR_GRAY2RGB)
    return imgOut

def erosion(img, thresholdval, ksize):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    n = 255
    retval, imB= cv2.threshold(img, thresholdval, n, cv2.THRESH_BINARY)  
    # kernel1 = np.ones((11,11), np.uint8)
    kernel2 = np.ones((ksize,ksize), np.uint8)
    # kernel3 = np.ones((45,45), np.uint8)
    # img_ero1 = cv2.erode(imB, kernel1, iterations=1)
    imgOut = cv2.erode(imB, kernel2, iterations=1)
    # img_ero3 = cv2.erode(imB, kernel3, iterations=1)
    imgOut = cv2.cvtColor(imgOut, cv2.COLOR_GRAY2RGB)
    return imgOut

def dilation(img, thresholdval, ksize):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    n = 255
    retval, imB= cv2.threshold(img, thresholdval, n, cv2.THRESH_BINARY)
    kernel2 = np.ones((ksize,ksize), np.uint8)
    imgOut = cv2.dilate(imB, kernel2, iterations=1)
    imgOut = cv2.cvtColor(imgOut, cv2.COLOR_GRAY2RGB)
    return imgOut

def boundaryExtraction(img, threshval= 150, ksize = 3):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    n = 255
    retval, imB = cv2.threshold(img, threshval, n, cv2.THRESH_BINARY) 
    kernel = np.ones((ksize, ksize), np.uint8)
    img_ero = cv2.erode(imB, kernel, iterations=1)
    imgReview = (img - img_ero)
    imgOut = 255 - imgReview
    imgOut = cv2.cvtColor(imgOut, cv2.COLOR_GRAY2RGB)
    return imgOut

def regionFilling(img, x, y, ksize = 5):
    A = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, A = cv2.threshold (A, 0, 1, cv2.THRESH_BINARY) 
    Label=np.zeros(A.shape)
    B = np.ones((ksize,ksize), 'uint8')
    X1=np.zeros(A.shape, 'uint8')
    X0=A.copy()
    X1[x,y]=1
    while (np.sum(X0!=X1)>0):
        X0=X1.copy()
        X1=cv2.dilate(X0,B,iterations=1)&A
    Label[X1==1]=1
    Label = (A - Label) * 255
    # Label = cv2.convertScaleAbs(Label)
    Label = cv2.cvtColor(np.array(Label, dtype=np.uint8) , cv2.COLOR_GRAY2RGB)
    return Label