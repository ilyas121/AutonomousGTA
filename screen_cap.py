import numpy as np
import cv2
import time
import pyautogui
import math

from directkeys import PressKey, ReleaseKey, W, A, S, D

region = (0, 40, 800, 640)


def average_slope_intercept(lines):
    left_lines    = []  # (slope, intercept)
    left_weights  = []  # (length,)
    right_lines   = []  # (slope, intercept)
    right_weights = []  # (length,)
    
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue  # ignore a vertical line
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                if slope < 0:  # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append(length)
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append(length)
        
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
        return (x1, y1), (x2, y2)
        
    except:
        # return garbage
        return (2,2), (3,3)


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    
    y1 = image.shape[0]  # bottom of the image
    y2 = y1*0.3          # slightly lower than the middle

    left_line  = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)
    
    return left_line, right_line

    
def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [0, 0, 255], 3)
    except:
        pass


def draw_lanes(img, lines):
    # print(lines)
    try:
        for line in list(lines):
            # print("Coords" + str(line))
            coords = line
            cv2.line(img, line[0], line[1], [255, 0, 0], 3)
    except:
        pass


def display_image(image_to_display):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1600, 900)

    cv2.imshow('image', image_to_display)
    cv2.waitKey()

    cv2.destroyAllWindows()


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_image(original_image):
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    processed_img = cv2.GaussianBlur(processed_img, (13, 13), 0)
    processed_img = cv2.Canny(processed_img, threshold1=30, threshold2=150)
    # vertices = np.array([[-580,1120], [640,575], [1280,575], [2500, 1120]])
    vertices = np.array([[0, 860], [840, 480], [1080, 480], [1920, 860]])
    processed_img = roi(processed_img, [vertices])

    return processed_img


def get_lines(img):
    return cv2.HoughLinesP(img, 1, np.pi / 180, 180, np.array([]), 150, 5)


# Whether or not to actually trigger keystrokes
do_key_presses = False


def turn_left():
    # Turn left
    print("left")
    print()
    if do_key_presses: ReleaseKey(D)
    if do_key_presses: PressKey(A)


def turn_right():
    # Turn right
    print("right")
    print()
    if do_key_presses: ReleaseKey(A)
    if do_key_presses: PressKey(D)


def turn_straight():
    # Don't turn
    print("straight")
    print()
    if do_key_presses: ReleaseKey(A)
    if do_key_presses: ReleaseKey(D)


def handle_image(image):
    processed_image = process_image(np.array(image))
    lines = get_lines(processed_image)
    lanes = lane_lines(processed_image, lines)

    draw_lines(processed_image, lines)
    draw_lanes(processed_image, lanes)

    lane1 = lanes[0]
    lane2 = lanes[1]

    slope1 = 0
    slope2 = 0

    xintercept1 = 960
    xintercept2 = 960

    avgx1 = 960
    avgx2 = 960

    left_lane = None
    right_lane = None
    left_slope = 0
    right_slope = 0

    if lane1 is not None:
        l1x1, l1y1, l1x2, l1y2 = lane1[0][0], lane1[0][1], lane1[1][0], lane1[1][1]

        slope1 = (l1y2 - l1y1) / (l1x2 - l1x1)
        yintercept1 = l1y1 - slope1 * l1x1
        xintercept1 = yintercept1 / slope1
        topintercept1 = xintercept1 + (1120 / slope1)
        avgx1 = (xintercept1 + topintercept1) / 2

    if lane2 is not None:
        l2x1, l2y1, l2x2, l2y2 = lane2[0][0], lane2[0][1], lane2[1][0], lane2[1][1]

        slope2 = (l2y2 - l2y1) / (l2x2 - l2x1)
        yintercept2 = l2y1 - slope2 * l2x1
        xintercept2 = yintercept2 / slope2
        topintercept2 = xintercept2 + (1120 / slope2)
        avgx2 = (xintercept2 + topintercept2) / 2

    # Get Lanes
    if xintercept1 < 960:
        left_lane = lane1
        left_slope = slope1
    elif xintercept2 < 960:
        left_lane = lane2
        left_slope = slope2

    if xintercept1 > 960:
        right_lane = lane1
        right_slope = slope1
    elif xintercept2 > 960:
        right_lane = lane2
        right_slope = slope2

    slope_tolerance = math.tan(20)
    tolerance = 0.2

    print("Slope1: {}".format(abs(slope1)))
    print("AvgX1: {}".format(avgx1))
    print("Slope2: {}".format(abs(slope2)))
    print("AvgX2: {}".format(avgx2))

    if 960 - (960 * tolerance) < avgx1 < 960 or 960 - (960 * tolerance) < avgx2 < 960\
            or abs(left_slope) < slope_tolerance:
        turn_right()

    elif 960 + (960 * tolerance) > avgx1 > 960 or 960 + (960 * tolerance) > avgx2 > 960\
            or abs(right_slope) < slope_tolerance:
        turn_left()

    else:
        turn_straight()

    display_image(processed_image)


def run_screen_capture(region):
    while True:
        start = time.time()
        image = pyautogui.screenshot(region=region)
        print("Time took: {}".format(time.time() - start))

        handle_image(image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1600, 900)

    image = cv2.imread('car3.PNG')
    handle_image(image)
    image = cv2.imread('car4.PNG')
    handle_image(image)
    image = cv2.imread('car5.PNG')
    handle_image(image)

    cv2.waitKey()
    cv2.destroyAllWindows()

    # PressKey(W)
    # region = (0, 40, 1920, 1120)
    # run_screen_capture(region)


