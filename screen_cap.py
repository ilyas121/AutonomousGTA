import numpy as np
import cv2
import time
import pyautogui
import argparse

region = (0,40,800,640)

def average_slope_intercept(lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2==x1:
                    continue # ignore a vertical line
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                if slope < 0: # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
        
        # add more weight to longer lines    
        left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
        right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
        
        return left_lane, right_lane # (slope, intercept), (slope, intercept)
    except:
        return (1,1), (1,1)

def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    try:
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        return ((x1, y1), (x2, y2))
        
    except:
        #return garbage
        return ((2,2), (3,3))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    
    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.3         # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    
    return left_line, right_line

    
def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [0,0,255], 3)
    except:
        pass

def draw_lanes(img, lines):
    print(lines)
    try:
        for line in list(lines):
            print("Coords" + str(line))
            coords = line
            cv2.line(img, line[0], line[1], [255,0, 0], 3)
    except:
       pass

def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([125, 125, 125])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([100, 100,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def select_white_yellow(image):
    #converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def get_running_average():
    pass

def process_image(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_RGB2HLS)
    #processed_img = select_rgb_white_yellow(processed_img)
    processed_img = cv2.GaussianBlur(processed_img, (19,19), 0)
    # rocessed_img = cv2.Canny(processed_img, threshold1 = 30, threshold2=150)p
    processed_img = auto_canny(processed_img);
    vertices = np.array([[130,1131], [672,564], [1278,573], [2008, 1115]])
    processed_img = roi(processed_img, [vertices] )
    #lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 150, 5)
    lines =  cv2.HoughLinesP(processed_img, rho=1, theta=np.pi/180, threshold=180, minLineLength=20, maxLineGap=300)
    lanes = lane_lines(processed_img, lines)
    draw_lines(processed_img, lines)

    draw_lanes(processed_img, lanes)
    return processed_img

def run_screen_capture(region):
    while(True):
        start = time.time()
        image = pyautogui.screenshot(region=region)
        print("Time took: {}".format(time.time() - start))
        
        cv2.imshow('image', process_image(np.array(image)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)
    cv2.imshow('image', process_image(cv2.imread('car4.PNG')))
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    region = (0,40, 1920, 1120)
    run_screen_capture(region)
    '''
