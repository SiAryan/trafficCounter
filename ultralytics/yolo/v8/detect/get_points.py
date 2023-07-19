import cv2
import ctypes

# Get the screen resolution
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

# Load the image
image_path = "GH016397_frame.jpg"
image = cv2.imread(image_path)

# Calculate the new size while maintaining the aspect ratio
aspect_ratio = image.shape[1] / image.shape[0]
new_width = min(screen_width, image.shape[1])
new_height = int(new_width / aspect_ratio)

# Resize the image
image = cv2.resize(image, (new_width, new_height))
# Create a named window to display the image
window_name = "Draw Line"
cv2.namedWindow(window_name)

# Variables to store the line coordinates
start_point = None
end_point = None

# Mouse callback function to capture the line coordinates
def mouse_callback(event, x, y, flags, param):
    global start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)

        # Draw the line on the image
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow(window_name, image)

# Set the mouse callback function for the window
cv2.setMouseCallback(window_name, mouse_callback)

# Display the image and wait for user input
cv2.imshow(window_name, image)
cv2.waitKey(0)

# Save the start and end points as variables
start_x, start_y = start_point
end_x, end_y = end_point

# Print the start and end points
print("Start point:", start_x, start_y)
print("End point:", end_x, end_y)

# Cleanup
cv2.destroyAllWindows()