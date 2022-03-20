# import the required libraries
import cv2
import numpy as np

#display the lane lines on the image
def display_lines(image, lines): 
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0), 10)
    return line_image

# turns the image into a gradient image 
def img_modify(image):
    lane_img = np.copy(image)
    gray = cv2.cvtColor(lane_img, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    return canny


#creates x, y coordinates out of a given slope and  y intercept
def make_coordinates(image, line_parameters): 
        slope, intercept = line_parameters
        y1 =  image.shape[0]
        y2 = int(y1*3/5)
        x1 = int((y1-intercept)/slope)
        x2 = int((y2-intercept)/slope)
        return np.array([x1,y1,x2,y2])

 # isolates the region where the lane lies in an image
def region_of_interest(image):  
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

 #Averages the slope of similar parallel lines in the image
def average_lines(image, lines):
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope>0:
                left_lines.append([slope, intercept])
            else:
                right_lines.append([slope, intercept])
        
        left_param = np.average(left_lines, axis=0)
        
        right_param = np.average(right_lines, axis=0)
        print(left_param, "lp")
        left_line = make_coordinates(image, left_param)
        if right_lines ==[]:
            return np.array([left_line])
        right_line = make_coordinates(image, right_param)
        return np.array([left_line, right_line])


cap = cv2.VideoCapture("test2.mp4")
while (cap.isOpened()):
    _, image = cap.read()
    mod_image = img_modify(image)
    area = region_of_interest(mod_image)
    lines = cv2.HoughLinesP(area, 2, np.pi/180,100, np.array([]), minLineLength=100, maxLineGap=5)
    final_lines = average_lines(image, lines)
    final_img = display_lines(image, final_lines)
    merge_img = cv2.addWeighted(image, 0.8, final_img, 1, 1)
    cv2.imshow("image", merge_img)
    # to exit the video in between press 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
