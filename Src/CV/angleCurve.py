 # Import necessary libraries
import cv2
import numpy as np

# Function to preprocess the image to detect yellow and white lanes
def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gblur = cv2.GaussianBlur(gray, (5, 5), 0)
    white_mask = cv2.threshold(gblur, 200, 255, cv2.THRESH_BINARY)[1]
    lower_yellow = np.array([0, 100, 100])
    upper_yellow = np.array([210, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return mask

# Function that defines the polygon region of interest
def regionOfInterest(img, polygon):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, np.int32([polygon]), 1)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

# Function that warps the image
def warp(img, source_points, destination_points, destn_size):
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    warped_img = cv2.warpPerspective(img, matrix, destn_size)
    return warped_img

# Function that unwarps the image
def unwarp(img, source_points, destination_points, source_size):
    matrix = cv2.getPerspectiveTransform(destination_points, source_points)
    unwarped_img = cv2.warpPerspective(img, matrix, source_size)
    return unwarped_img

# Function that gives the left fit and right fit curves for the lanes in bird's-eye view
def fitCurve(img):
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 50
    margin = 100
    minpix = 50
    window_height = int(img.shape[0]/nwindows)
    y, x = img.nonzero()
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_indices = []
    right_lane_indices = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window+1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_indices = ((y >= win_y_low) & (y < win_y_high) &
                             (x >= win_xleft_low) & (x < win_xleft_high)).nonzero()[0]
        good_right_indices = ((y >= win_y_low) & (y < win_y_high) &
                              (x >= win_xright_low) & (x < win_xright_high)).nonzero()[0]
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)
        if len(good_left_indices) > minpix:
            leftx_current = int(np.mean(x[good_left_indices]))
        if len(good_right_indices) > minpix:
            rightx_current = int(np.mean(x[good_right_indices]))

    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)
    leftx = x[left_lane_indices]
    lefty = y[left_lane_indices]
    rightx = x[right_lane_indices]
    righty = y[right_lane_indices]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

# Function to calculate turn angle based on lane center deviation
def calculateTurnAngle(img, left_fit, right_fit):
    y_eval = img.shape[0]
    left_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
    right_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
    lane_center = (left_x + right_x) / 2
    image_center = img.shape[1] / 2
    angle_offset = np.arctan((lane_center - image_center) / y_eval) * (180 / np.pi)
    return round(angle_offset, 2)

# Function to display angle information on the result image
def displayTurnAngle(img, angle_offset):
    turn_direction = 'Left' if angle_offset < 0 else 'Right'
    text = f'Turn {turn_direction} by {abs(angle_offset)} degrees'
    img = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return img

# Function that predicts turn based on radius
def addTurnInfo(img, radius):
    if radius >= 10000:
        img = cv2.putText(img, 'Go Straight', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    elif radius >= 0 and radius < 10000:
        img = cv2.putText(img, 'Turn Right', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        img = cv2.putText(img, 'Turn Left', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return img

# Main video processing
video = cv2.VideoCapture(0)
out = cv2.VideoWriter('results/curve_lane_detection_with_angle.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (1280, 720))

while True:
    isTrue, frame = video.read()
    if not isTrue:
        break

    processed_img = preprocessing(frame)
    height, width = processed_img.shape
    polygon = [
        (int(width * 0.15), int(height * 0.94)),
        (int(width * 0.45), int(height * 0.62)),
        (int(width * 0.58), int(height * 0.62)),
        (int(0.95 * width), int(0.94 * height))
    ]
    masked_img = regionOfInterest(processed_img, polygon)
    source_points = np.float32([
        [int(width * 0.49), int(height * 0.62)],
        [int(width * 0.58), int(height * 0.62)],
        [int(width * 0.15), int(height * 0.94)],
        [int(0.95 * width), int(0.94 * height)]
    ])
    destination_points = np.float32([[0, 0], [400, 0], [0, 960], [400, 960]])
    warped_img_size = (400, 960)
    warped_img_shape = (960, 400)
    warped_img = warp(masked_img, source_points, destination_points, warped_img_size)
    left_fit, right_fit = fitCurve(warped_img)
    turn_angle = calculateTurnAngle(warped_img, left_fit, right_fit)
    result = displayTurnAngle(frame, turn_angle)

    # Show and save output
    cv2.imshow("Lane Detection", result)
    out.write(result)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()
print("Video output generated with turn angle information.\n")