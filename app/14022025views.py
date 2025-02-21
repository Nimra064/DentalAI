
# Detection Updated 

# import os
# import cv2
# import base64
# import numpy as np
# from collections import defaultdict
# from django.http import JsonResponse
# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser, FormParser
# from app.models import UploadedImage  # Ensure this model is correctly defined
# from ultralytics import YOLO  # Ensure ultralytics package is installed

# # Function to calculate Intersection over Union (IoU)
# def calculate_iou(box1, box2):
#     x1, y1, x2, y2 = box1
#     x1_p, y1_p, x2_p, y2_p = box2

#     xi1 = max(x1, x1_p)
#     yi1 = max(y1, y1_p)
#     xi2 = min(x2, x2_p)
#     yi2 = min(y2, y2_p)

#     inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (x2_p - x1_p) * (y2_p - y1_p)

#     return inter_area / float(box1_area + box2_area - inter_area)

# class ImageUploadView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request, *args, **kwargs):
#         try:
#             uploaded_image = request.FILES['image']
#             image_instance = UploadedImage.objects.create(image=uploaded_image)
#             image_path = image_instance.image.path
#             original_image = cv2.imread(image_path)

#             model1 = YOLO('app/number.pt')
#             model2 = YOLO('app/detect.pt')

#             object_id_counter = 0
#             object_tracker = defaultdict(lambda: None)
#             IOU_THRESHOLD = 0.5

#             results1 = model1(image_path, conf=0.25)

#             final_detections = defaultdict(lambda: {"Tooth #": "", "tooth_area_in_pixels": "", "pulp_area_in_pixels": ""})
#             results_folder = os.path.join('static', 'results')
#             os.makedirs(results_folder, exist_ok=True)

#             for result1 in results1:
#                 for box, label, conf1 in zip(result1.boxes.xyxy, result1.boxes.cls, result1.boxes.conf):
#                     x_min, y_min, x_max, y_max = map(int, box)
#                     class_label = model1.names[int(label)]
#                     object_id_counter += 1
#                     object_tracker[tuple(box)] = object_id_counter

#                     cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                     label_text = f"{class_label}: {conf1:.2f}"
#                     cv2.putText(original_image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                                 0.5, (0, 255, 0), 2)
                    
#                     cropped_region = original_image[y_min:y_max, x_min:x_max]
#                     results2 = model2(cropped_region, conf=0.25)

#                     tooth_area, pulp_area = "None", "None"
                    
#                     for result2 in results2:
#                         for box2, label2, conf2 in zip(result2.boxes.xyxy, result2.boxes.cls, result2.boxes.conf):
#                             x_min2, y_min2, x_max2, y_max2 = map(int, box2)
#                             class_label2 = model2.names[int(label2)]

#                             cv2.rectangle(cropped_region, (x_min2, y_min2), (x_max2, y_max2), (255, 0, 0), 3)
#                             label_text2 = f"{class_label2}: {conf2:.2f}"
#                             cv2.putText(cropped_region, label_text2, (x_min2, y_min2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                                         0.5, (255, 0, 0), 2)
            
#                             if class_label2 == "tooth":
#                                 tooth_area = (x_max2 - x_min2) * (y_max2 - y_min2)
#                             elif class_label2 == "pulp":
#                                 pulp_area = (x_max2 - x_min2) * (y_max2 - y_min2)
                    
#                     final_detections[object_id_counter]["Tooth #"] += f"{class_label}, "
#                     final_detections[object_id_counter]["tooth_area_in_pixels"] = tooth_area
#                     final_detections[object_id_counter]["pulp_area_in_pixels"] = pulp_area
            
#             annotated_image_path = os.path.join(results_folder, f"annotated_{os.path.basename(image_path)}")
#             cv2.imwrite(annotated_image_path, original_image)

#             with open(annotated_image_path, "rb") as image_file:
#                 base64_image = base64.b64encode(image_file.read()).decode('utf-8')

#             response_detections = []
#             for object_id, detection in final_detections.items():
#                 tooth_number = detection["Tooth #"].split('_')[-1].rstrip(', ') if detection["Tooth #"] else ""
#                 detection["Tooth #"] = tooth_number
#                 response_detections.append(detection)

#             grouped_detections = []
#             last_detection = None

#             for detection in response_detections:
#                 if last_detection is None:
#                     last_detection = detection
#                 else:
#                     if last_detection["Tooth #"] == detection["Tooth #"]:
#                         last_detection["tooth_area_in_pixels"] = detection["tooth_area_in_pixels"]
#                         last_detection["pulp_area_in_pixels"] = detection["pulp_area_in_pixels"]
#                     else:
#                         grouped_detections.append(last_detection)
#                         last_detection = detection

#             if last_detection:
#                 grouped_detections.append(last_detection)

#             response_data = {
#                 "StatusCode": 200,
#                 "AnnotatedImage": f"http://127.0.0.1:8000/{results_folder}/{os.path.basename(annotated_image_path)}",
#                 "Data": grouped_detections
#             }

#             return JsonResponse(response_data)

#         except Exception as e:
#             return JsonResponse({
#                 "StatusCode": 500,
#                 "Message": str(e)
#             }, status=500)






#seg


#import os
#import cv2
#import base64
#import numpy as np
#from collections import defaultdict
#from django.http import JsonResponse
#from rest_framework.views import APIView
#from rest_framework.parsers import MultiPartParser, FormParser
#from app.models import UploadedImage  # Ensure this model is correctly defined
#from ultralytics import YOLO  # Ensure ultralytics package is installed

#class ImageUploadView(APIView):
#    parser_classes = (MultiPartParser, FormParser)

#   def post(self, request, *args, **kwargs):
 #       try:
  #          uploaded_image = request.FILES['image']
   #         image_instance = UploadedImage.objects.create(image=uploaded_image)
    #        image_path = image_instance.image.path
     #       original_image = cv2.imread(image_path)

      #      model1 = YOLO('app/model/number.pt')
       #     model2 = YOLO('app/model/detect.pt')

        #    object_id_counter = 0
         #   object_tracker = defaultdict(lambda: None)

          #  results1 = model1(image_path, conf=0.25)

           # final_detections = defaultdict(lambda: {"tooth_no": "", "tooth_area_in_pixels": "", "pulp_area_in_pixels": ""})
           # results_folder = os.path.join('static', 'results')
          #  os.makedirs(results_folder, exist_ok=True)
            
           # for result1 in results1:

            #    for box, label, conf1 in zip(result1.boxes.xyxy, result1.boxes.cls, result1.boxes.conf):
             #           x_min, y_min, x_max, y_max = map(int, box)
              #          class_label = model1.names[int(label)]
               #         object_id_counter += 1
                #        object_tracker[tuple(box)] = object_id_counter

                 #       cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                  #      label_text = f"{class_label}: {conf1:.2f}"
                   #     cv2.putText(original_image, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    #                0.5, (0, 255, 0), 2)
                        
                     #   cropped_region = original_image[y_min:y_max, x_min:x_max]
                      #  results2 = model2(cropped_region, conf=0.25)

                       # tooth_area, pulp_area = "None", "None"
                        
                       # for result2 in results2:
                        #    for mask2, label2, conf2 in zip(result2.masks.xy, result2.boxes.cls, result2.boxes.conf):
                         #       class_label2 = model2.names[int(label2)]
                          #      polygon2 = np.array(mask2, np.int32)

                           #     fill_color = (255,127,127) if class_label2 == "tooth" else (144, 239, 144) 
                            #    cv2.fillPoly(cropped_region, [polygon2], fill_color)
                                
                             #   if class_label2 == "tooth":
                              #      tooth_area = cv2.contourArea(polygon2)
                              #  elif class_label2 == "pulp":
                               #     pulp_area = cv2.contourArea(polygon2)
                        
                       # final_detections[object_id_counter]["tooth_no"] += f"{class_label}, "
                       # final_detections[object_id_counter]["tooth_area_in_pixels"] = tooth_area
                       # final_detections[object_id_counter]["pulp_area_in_pixels"] = pulp_area
            
           # annotated_image_path = os.path.join(results_folder, f"annotated_{os.path.basename(image_path)}")
           # cv2.imwrite(annotated_image_path, original_image)

           # with open(annotated_image_path, "rb") as image_file:
            #    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

           # response_detections = []
          #  for object_id, detection in final_detections.items():
           #     tooth_number = detection["tooth_no"].split('_')[-1].rstrip(', ') if detection["tooth_no"] else ""
            #    detection["tooth_no"] = tooth_number
             #   response_detections.append(detection)

           # grouped_detections = []
           # last_detection = None

           # for detection in response_detections:
            #    if last_detection is None:
             #       last_detection = detection
              #  else:
               #     if last_detection["tooth_no"] == detection["tooth_no"]:
                #        last_detection["tooth_area_in_pixels"] = detection["tooth_area_in_pixels"]
                 #       last_detection["pulp_area_in_pixels"] = detection["pulp_area_in_pixels"]
                  #  else:
                   #     grouped_detections.append(last_detection)
                    #    last_detection = detection

           # if last_detection:
            #    grouped_detections.append(last_detection)

           # response_data = {
            #    "StatusCode": 200,
             #   "Image Path": f"http://analysis-api-v2.dentalid.app/{results_folder}/{os.path.basename(annotated_image_path)}",
              #  "Data": grouped_detections
           # }

           # return JsonResponse(response_data)

       # except Exception as e:
        #    return JsonResponse({
         #       "StatusCode": 500,
          #      "Message": str(e)
           # }, status=500)
import os
import cv2
import base64
import numpy as np
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
            model1 = YOLO('app/model/number.pt')  # Model for Tooth Number Detection
            model2 = YOLO('app/model/detect.pt')  # Model for Tooth & Pulp Area Detection

            results1 = model1(image_path, conf=0.25)
            results2 = model2(image_path, conf=0.25)

            detections = []  
            tooth_number_map = {}  

            # **Step 1: Run model1 (Tooth Number Detection)**
            if results1 and results1[0].boxes is not None:
                for box in results1[0].boxes.data:
                    x_min, y_min, x_max, y_max, conf1, label = map(int, box.tolist())
                    class_label = model1.names[label]  # Get the tooth number label
                    tooth_number_map[(x_min, y_min, x_max, y_max)] = class_label  

                    # Draw bounding box for tooth number
                    cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(original_image, f"{class_label}", (x_min, y_min - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # **Step 2: Run model2 (Tooth & Pulp Area Detection)**
            if results2 and results2[0].masks is not None:
                overlay = original_image.copy()  # Create an overlay for transparency
                for mask2, box2 in zip(results2[0].masks.xy, results2[0].boxes.data):
                    x_min, y_min, x_max, y_max, conf2, label2 = map(int, box2.tolist())
                    class_label2 = model2.names[label2]

                    polygon2 = np.array(mask2, np.int32)

                    # Define colors for tooth and pulp areas
                    fill_color = (255, 182, 193) if class_label2 == "tooth" else (173, 216, 230)  

                    # **Ensure all coordinates are filled properly using anti-aliasing**
                    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [polygon2], 255)
                    
                    # Use Gaussian blur for better anti-aliasing
                    mask = cv2.GaussianBlur(mask, (5, 5), 0)
                    mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1]

                    # Apply the mask to the overlay
                    for c in range(3):  
                        overlay[:, :, c] = np.where(mask == 255, fill_color[c], overlay[:, :, c])

                    assigned_tooth_number = ""
                    for (tx_min, ty_min, tx_max, ty_max), tooth_number in tooth_number_map.items():
                        if (tx_min < polygon2[:, 0].mean() < tx_max) and (ty_min < polygon2[:, 1].mean() < ty_max):
                            assigned_tooth_number = tooth_number
                            break

                    assigned_tooth_number = assigned_tooth_number.split("_")[-1]

                    # Compute the polygon area
                    area_pixels = cv2.contourArea(polygon2)
                    if class_label2 == "tooth":
                        detections.append({
                            "tooth_no": assigned_tooth_number,  
                            "tooth_area_in_pixels": area_pixels,
                            "pulp_area_in_pixels": 0  # Placeholder
                        })
                    elif class_label2 == "pulp":
                        for item in detections:
                            if item["tooth_no"] == assigned_tooth_number:
                                item["pulp_area_in_pixels"] = area_pixels

                # Blend overlay with transparency effect
                alpha = 0.6  # Adjust transparency (0 = fully transparent, 1 = opaque)
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
