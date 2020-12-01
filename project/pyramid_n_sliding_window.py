import numpy as np
from skimage.feature import hog
from skimage.transform import rescale, resize

def pyramid(image, scale, minSize=(50, 50)):
    x,y = image.shape
    if (x>300) or (y>300):
        image = rescale(image, 0.6)
    images = []
    current_scale = 1.0
    images.append((current_scale, image))
    while current_scale * image.shape[0] > minSize[0] and current_scale * image.shape[1] > minSize[1]:
        current_scale *= scale
        images.append((current_scale, rescale(image, current_scale)))

    return images


def sliding_window(image, base_score, stepSize, windowSize, pixel_per_cell=8):
    (max_score, maxr, maxc) = (0,0,0)
    winH, winW = windowSize
    H,W = image.shape
    pad_image = np.lib.pad(image, ((winH//2,winH-winH//2),(winW//2, winW-winW//2)), mode='constant')
    for i in range(0, H+1, stepSize):
        for j in range(0, W+1, stepSize):
            #hogFeature, hogImage = hog_feature(pad_image[i][j])
            window = pad_image[i:i+winH, j:j+winW]
            temp_hog = hog(window)
            
            result = np.dot(temp_hog, base_score)                
            if result > max_score:
                max_score = result
                maxr = i-winH//2
                maxc = j-winW//2
    
    return max_score, maxr, maxc


def pyramid_score(image, base_score, windowSize, stepSize=20, scale = 0.9):
    max_score = 0
    maxr = 0
    maxc = 0
    max_scale = 1.0
    images = pyramid(image, scale)
    for s, i in images:
        score, r, c = sliding_window(i, base_score, stepSize, windowSize, pixel_per_cell=8)
        if score > max_score:
            max_score = score
            maxr = r
            maxc = c
            max_scale = s
    return max_score, maxr, maxc, max_scale

# def pyramid(image, scale=0.9, minSize=(100, 100)):

#     # yield the original image
#     images = []
#     current_scale = 1.0
#     images.append((current_scale, image))
#     # keep looping over the pyramid
#     #####################################
#     #       START YOUR CODE HERE        #
#     #####################################
#     while current_scale * image.shape[0] > minSize[0] and current_scale * image.shape[1] > minSize[1]:
#         current_scale *= scale
#         images.append((current_scale, rescale(image, current_scale)))
#     ######################################
#     #        END OF YOUR CODE            #
#     ######################################
#     return images

# def sliding_window(image, base_score, stepSize, windowSize, pixel_per_cell=8):

#     # slide a window across the image
#     (max_score, maxr, maxc) = (0,0,0)
#     winH, winW = windowSize
#     H,W = image.shape
#     pad_image = np.lib.pad(image, ((winH//2,winH-winH//2),(winW//2, winW-winW//2)), mode='constant')
#     response_map = np.zeros((H//stepSize+1, W//stepSize+1))
    
#     #####################################
#     #       START YOUR CODE HERE        #
#     #####################################
#     for i in range(0, H+1, stepSize):
#         for j in range(0, W+1, stepSize):
#             #hogFeature, hogImage = hog_feature(pad_image[i][j])
#             window = pad_image[i:i+winH, j:j+winW]
#             temp_hog = hog(window)
            
#             result = np.dot(temp_hog, base_score)                
#             if result > max_score:
#                 max_score = result
#                 maxr = i-winH//2
#                 maxc = j-winW//2
#             if i==0 and j==0:
#                 response_map[i, j] = result
#             elif i==0 and j>0:
#                 response_map[i, int(j//stepSize)] = result
#             elif i>0 and j==0:
#                 response_map[int(i//stepSize), j] = result
#             else:
#                 response_map[int(i//stepSize), int(j//stepSize)] = result

#     ######################################
#     #        END OF YOUR CODE            #
#     ######################################
    
    
#     return (max_score, maxr, maxc, response_map)

# def pyramid_score(image, base_score, shape, stepSize=20, scale = 0.9, pixel_per_cell = 8):

#     max_score = 0
#     maxr = 0
#     maxc = 0
#     max_scale = 1.0
#     max_response_map =np.zeros(image.shape)
#     images = pyramid(image, scale)
#     #####################################
#     #       START YOUR CODE HERE        #
#     #####################################
#     images = pyramid(image, scale=scale)
#     for s, i in images:
#         score, r, c, m = sliding_window(i, base_score, stepSize, shape, pixel_per_cell=8)
#         if score > max_score:
#             max_score = score
#             maxr = r
#             maxc = c
#             max_response_map = m
#             max_scale = s
#     ######################################
#     #        END OF YOUR CODE            #
#     ######################################
#     c