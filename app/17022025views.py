import os
import cv2
import base64
import numpy as np
import random
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from app.models import UploadedImage
from ultralytics import YOLO  

class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            uploaded_image = request.FILES['image']
            image_instance = UploadedImage.objects.create(image=uploaded_image)
            image_path = image_instance.image.path
            original_image = cv2.imread(image_path)

            # Load YOLO models
            model1 = YOLO('app/model/number.pt')  # Tooth Number Detection
            model2 = YOLO('app/model/detect.pt')  # Tooth & Pulp Area Detection

            results1 = model1(image_path, conf=0.25)
            results2 = model2(image_path, conf=0.25)

            detections = []  
            tooth_number_map = {}  
            tooth_counter = 1  # Counter for tooth labels (T1, T2, ...)

            # **Step 1: Run model1 (Tooth Number Detection)**
            if results1 and results1[0].boxes is not None:
                for box in results1[0].boxes.data:
                    x_min, y_min, x_max, y_max, conf1, label = map(int, box.tolist())
                    class_label = model1.names[label]  
                    tooth_number_map[(x_min, y_min, x_max, y_max)] = class_label  

                    # Draw bounding box for tooth number
                  #  cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                   # cv2.putText(original_image, f"{class_label}", (x_min, y_min - 10), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # **Step 2: Run model2 (Tooth & Pulp Area Detection)**
            if results2 and results2[0].masks is not None:
                overlay = original_image.copy()  
                instance_colors = {}  

                for idx, (mask2, box2) in enumerate(zip(results2[0].masks.xy, results2[0].boxes.data)):
                    x_min, y_min, x_max, y_max, conf2, label2 = map(int, box2.tolist())
                    class_label2 = model2.names[label2]

                    polygon2 = np.array(mask2, np.int32)

                    if idx not in instance_colors:
                        instance_colors[idx] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                    fill_color = instance_colors[idx]  

                    # Apply segmentation mask
                    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [polygon2], 255)
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)
                    mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]

                    for c in range(3):  
                        overlay[:, :, c] = np.where(mask == 255, fill_color[c], overlay[:, :, c])

                    assigned_tooth_number = ""
                    for (tx_min, ty_min, tx_max, ty_max), tooth_number in tooth_number_map.items():
                        if (tx_min < polygon2[:, 0].mean() < tx_max) and (ty_min < polygon2[:, 1].mean() < ty_max):
                            assigned_tooth_number = tooth_number
                            break

                    assigned_tooth_number = assigned_tooth_number.split("_")[-1]

                    area_pixels = cv2.contourArea(polygon2)

                    if class_label2 == "tooth":
                        t_label = f"t{tooth_counter}"  # Assign T1, T2, ...
                        tooth_counter += 1  

                        # **Fix: Center the Label Inside the Tooth**
                        center_x = int(np.mean(polygon2[:, 0]))  
                        center_y = int(np.mean(polygon2[:, 1]))  

                        (text_width, text_height), _ = cv2.getTextSize(t_label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 5)

                        # **Position Label at the Center of the Tooth**
                        label_x = center_x - text_width // 2
                        label_y = center_y + text_height // 2  

                        # # Draw black outline for better readability
                        # cv2.putText(original_image, t_label, 
                        #             (label_x, label_y),  
                        #             cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8, cv2.LINE_AA)  

                        # # Draw slightly darker white text on top (Light Gray)
                        # cv2.putText(original_image, t_label, 
                        #             (label_x, label_y),  
                        #             cv2.FONT_HERSHEY_SIMPLEX, 3, (256, 256, 256), 5, cv2.LINE_AA)  
  
                        # Draw black outline for better readability
                        cv2.putText(original_image, t_label, 
                                    (label_x, label_y),  
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)  

                        # Draw light gray text slightly inside the black outline
                        cv2.putText(original_image, t_label, 
                                    (label_x, label_y),  
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (200, 200, 200), 6, cv2.LINE_AA)  

                        # Draw final white text on top
                        cv2.putText(original_image, t_label, 
                                    (label_x, label_y),  
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3, cv2.LINE_AA)  


                        detections.append({
                            "tooth_no": assigned_tooth_number,  
                            "tooth_area_in_pixels": area_pixels,
                            "pulp_area_in_pixels": 0  
                        })

                    elif class_label2 == "pulp":
                        for item in detections:
                            if item["tooth_no"] == assigned_tooth_number:
                                item["pulp_area_in_pixels"] = area_pixels

                # Blend overlay with transparency effect
                alpha = 0.6  
                original_image = cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0)

            # **Step 3: Save and Return Response**
            results_folder = os.path.join('static', 'results')
            os.makedirs(results_folder, exist_ok=True)
            annotated_image_path = os.path.join(results_folder, f"annotated_{os.path.basename(image_path)}")
            cv2.imwrite(annotated_image_path, original_image)

            with open(annotated_image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            response_data = {
                "StatusCode": 200,
                "Image Path": f"https://analysis-api-v2.dentalid.app/{annotated_image_path.replace(os.sep, '/')}",
                "Data": detections
            }

            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({
                "StatusCode": 500,
                "Message": str(e)
            }, status=500)
