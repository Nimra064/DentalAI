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
