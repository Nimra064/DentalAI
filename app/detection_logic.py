import cv2
import numpy as np

def detect_objects_and_calculate_areas(image_path):
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {"error": "Image not found or invalid"}

    # Apply threshold to detect tooth and pulp
    _, tooth_mask = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    _, pulp_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # Find contours for tooth
    contours_tooth, _ = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tooth_areas = [cv2.contourArea(contour) for contour in contours_tooth]

    # Find contours for pulp
    contours_pulp, _ = cv2.findContours(pulp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pulp_areas = [cv2.contourArea(contour) for contour in contours_pulp]

    # Match areas to teeth
    data = []
    for tooth_area in tooth_areas:
        pulp_area = min(pulp_areas, default=0)  # Match closest pulp area (simplified logic)
        data.append({
            "tooth_area_in_pixels": str(int(tooth_area)),
            "pulp_area_in_pixels": str(int(pulp_area)) if pulp_area else "None"
        })
        if pulp_area:
            pulp_areas.remove(pulp_area)

    return {"status": "success", "data": data}


# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Calculate the intersection area
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate the union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    # Return IoU
    return inter_area / union_area if union_area > 0 else 0