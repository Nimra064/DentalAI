# # # # # # from django.http import JsonResponse
# # # # # # from rest_framework.views import APIView
# # # # # # from rest_framework.parsers import MultiPartParser, FormParser
# # # # # # from .models import UploadedImage
# # # # # # from ultralytics import YOLO
# # # # # # import cv2

# # # # # # class ImageUploadView(APIView):
# # # # # #     parser_classes = (MultiPartParser, FormParser)

# # # # # #     def post(self, request, *args, **kwargs):
# # # # # #         # Save the uploaded image
# # # # # #         uploaded_image = request.FILES['image']
# # # # # #         image_instance = UploadedImage.objects.create(image=uploaded_image)

# # # # # #         # Load the YOLO model
# # # # # #         model = YOLO('app/trainingmodel/best.pt')

# # # # # #         # Perform inference on the uploaded image
# # # # # #         image_path = image_instance.image.path
# # # # # #         results = model(image_path)

# # # # # #         # Extract labels of detected objects
# # # # # #         detected_labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
        
# # # # # #         print(detected_labels)
# # # # # #         # Return detected labels as JSON response
# # # # # #         return JsonResponse({
# # # # # #             'status': 'success',
# # # # # #             'detected_labels': detected_labels
# # # # # #         })


# # # # # import os
# # # # # from django.http import JsonResponse
# # # # # from rest_framework.views import APIView
# # # # # from rest_framework.parsers import MultiPartParser, FormParser
# # # # # from .models import UploadedImage
# # # # # from ultralytics import YOLO
# # # # # import cv2

# # # # # class ImageUploadView(APIView):
# # # # #     parser_classes = (MultiPartParser, FormParser)

# # # # #     def post(self, request, *args, **kwargs):
# # # # #         try:
# # # # #             # Save the uploaded image
# # # # #             uploaded_image = request.FILES['image']
# # # # #             image_instance = UploadedImage.objects.create(image=uploaded_image)

# # # # #             # Load the YOLO model
# # # # #             model = YOLO('app/trainingmodel/best.pt')

# # # # #             # Perform inference on the uploaded image
# # # # #             image_path = image_instance.image.path
# # # # #             results = model(image_path)

# # # # #             # Create a folder for saving annotated images
# # # # #             results_folder = os.path.join('media', 'results')
# # # # #             os.makedirs(results_folder, exist_ok=True)

# # # # #             # Generate the annotated image
# # # # #             annotated_image_path = os.path.join(results_folder, f"annotated_{os.path.basename(image_path)}")
# # # # #             results[0].plot()  # Generate the annotated image
# # # # #             cv2.imwrite(annotated_image_path, results[0].plot())  # Save the annotated image

# # # # #             # Extract detected labels, confidence scores, and bounding boxes
# # # # #             detected_objects = []
# # # # #             for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
# # # # #                 detected_objects.append({
# # # # #                     "label": results[0].names[int(cls)],
# # # # #                     "confidence": float(conf),
# # # # #                     "bounding_box": box.tolist()
# # # # #                 })

# # # # #             # Return the response with image path, detected labels, and accuracy
# # # # #             return JsonResponse({
# # # # #                 'status': 'success',
# # # # #                 'image_path': annotated_image_path,
# # # # #                 'detected_objects': detected_objects
# # # # #             })

# # # # #         except Exception as e:
# # # # #             return JsonResponse({
# # # # #                 'status': 'error',
# # # # #                 'message': str(e)
# # # # #             }, status=500)



# # # # import os
# # # # from django.http import JsonResponse
# # # # from rest_framework.views import APIView
# # # # from rest_framework.parsers import MultiPartParser, FormParser
# # # # from .models import UploadedImage
# # # # from ultralytics import YOLO
# # # # import cv2

# # # # class ImageUploadView(APIView):
# # # #     parser_classes = (MultiPartParser, FormParser)

# # # #     def post(self, request, *args, **kwargs):
# # # #         try:
# # # #             # Save the uploaded image
# # # #             uploaded_image = request.FILES['image']
# # # #             image_instance = UploadedImage.objects.create(image=uploaded_image)

# # # #             # Load the YOLO model
# # # #             model = YOLO('app/trainingmodel/best.pt')

# # # #             # Perform inference on the uploaded image
# # # #             image_path = image_instance.image.path
# # # #             results = model(image_path)

# # # #             # Create a folder for saving annotated images
# # # #             results_folder = os.path.join('media', 'results')
# # # #             os.makedirs(results_folder, exist_ok=True)

# # # #             # Generate the annotated image
# # # #             annotated_image_path = os.path.join(results_folder, f"annotated_{os.path.basename(image_path)}")
# # # #             annotated_image = results[0].plot()  # Generate the annotated image
# # # #             cv2.imwrite(annotated_image_path, annotated_image)  # Save the annotated image

# # # #             # Extract detected labels, confidence scores, bounding boxes, and calculate area
# # # #             detected_objects = []
# # # #             for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
# # # #                 # Convert tensor to list
# # # #                 box = box.tolist()
# # # #                 x1, y1, x2, y2 = box

# # # #                 # Calculate area
# # # #                 width = x2 - x1
# # # #                 height = y2 - y1
# # # #                 area = width * height

# # # #                 detected_objects.append({
# # # #                     "label": results[0].names[int(cls)],
# # # #                     "confidence": float(conf.item()),  # Convert tensor to float
# # # #                     "bounding_box": box,  # Bounding box as list
# # # #                     "area": str(area) + "px" , # Area of the bounding box
# # # #                 })

# # # #             # Return the response with image path, detected labels, and accuracy
# # # #             return JsonResponse({
# # # #                 'status': 'success',
# # # #                 'image_path': annotated_image_path,
# # # #                 'detected_objects': detected_objects
# # # #             })

# # # #         except Exception as e:
# # # #             return JsonResponse({
# # # #                 'status': 'error',
# # # #                 'message': str(e)
# # # #             }, status=500)



# # # # from rest_framework.views import APIView
# # # # from rest_framework.response import Response
# # # # from rest_framework.parsers import MultiPartParser, FormParser
# # # # from .detection_logic import detect_objects_and_calculate_areas
# # # # import os

# # # # class ImageUploadView(APIView):
# # # #     parser_classes = (MultiPartParser, FormParser)

# # # #     def post(self, request, format=None):
# # # #         file = request.FILES.get('image')
# # # #         if not file:
# # # #             return Response({"error": "No file provided"}, status=400)

# # # #         # Save uploaded file
# # # #         image_path = f'temp_images/{file.name}'
# # # #         os.makedirs('temp_images', exist_ok=True)
# # # #         with open(image_path, 'wb+') as f:
# # # #             for chunk in file.chunks():
# # # #                 f.write(chunk)

# # # #         # Process image
# # # #         result = detect_objects_and_calculate_areas(image_path)
# # # #         os.remove(image_path)  # Clean up
# # # #         return Response(result)



# # # import os
# # # import cv2
# # # import numpy as np
# # # from django.http import JsonResponse
# # # from rest_framework.views import APIView
# # # from rest_framework.parsers import MultiPartParser, FormParser
# # # from .models import UploadedImage
# # # from ultralytics import YOLO

# # # class ImageUploadView(APIView):
# # #     parser_classes = (MultiPartParser, FormParser)

# # #     def post(self, request, *args, **kwargs):
# # #         try:
# # #             # Save the uploaded image
# # #             uploaded_image = request.FILES['image']
# # #             image_instance = UploadedImage.objects.create(image=uploaded_image)

# # #             # Load the YOLO model
# # #             model = YOLO('app/trainingmodel/best.pt')
            
 
# # #             # Perform inference on the uploaded image
# # #             image_path = image_instance.image.path
# # #             print(image_path)
# # #             results = model(image_path)
            
# # #             print(results)
# # #             # Create a folder for saving annotated images
# # #             results_folder = os.path.join('media', 'results')
# # #             os.makedirs(results_folder, exist_ok=True)

# # #             # Generate the annotated image
# # #             annotated_image_path = os.path.join(results_folder, f"annotated_{os.path.basename(image_path)}")
# # #             annotated_image = results[0].plot()  # Generate the annotated image
# # #             cv2.imwrite(annotated_image_path, annotated_image)  # Save the annotated image

# # #             # Extract detected labels, confidence scores, bounding boxes, and calculate area from masks
# # #             detected_objects = []
# # #             for i, (box, cls, conf, mask) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf, results[0].masks)):
# # #                 # Convert tensor to list
# # #                 box = box.tolist()
# # #                 x1, y1, x2, y2 = box

# # #                 # Calculate area using the segmentation mask
# # #                 mask_array = mask.numpy()  # Convert mask tensor to numpy array
# # #                 mask_area = np.sum(mask_array > 0)  # Count non-zero pixels in the mask

# # #                 detected_objects.append({
# # #                     "label": results[0].names[int(cls)],
# # #                     "confidence": float(conf.item()),  # Convert tensor to float
# # #                     "bounding_box": box,  # Bounding box as list
# # #                     "area": f"{mask_area}px",  # Area from the mask in pixels
# # #                 })

# # #             # Return the response with image path, detected labels, and accuracy
# # #             return JsonResponse({
# # #                 'status': 'success',
# # #                 'image_path': annotated_image_path,
# # #                 'detected_objects': detected_objects
# # #             })

# # #         except Exception as e:
# # #             return JsonResponse({
# # #                 'status': 'error',
# # #                 'message': str(e)
# # #             }, status=500)



# # # import os
# # # import cv2
# # # import numpy as np
# # # from django.http import JsonResponse
# # # from rest_framework.views import APIView
# # # from rest_framework.parsers import MultiPartParser, FormParser
# # # from .models import UploadedImage
# # # from ultralytics import YOLO

# # # class ImageUploadView(APIView):
# # #     parser_classes = (MultiPartParser, FormParser)

# # #     def post(self, request, *args, **kwargs):
# # #         try:
# # #             # Save the uploaded image
# # #             uploaded_image = request.FILES['image']
# # #             image_instance = UploadedImage.objects.create(image=uploaded_image)

# # #             # Load the YOLO model
# # #             model = YOLO('app/trainingmodel/best.pt')

# # #             # Perform inference on the uploaded image
# # #             image_path = image_instance.image.path
# # #             results = model(image_path)

# # #             # Create a folder for saving annotated images
# # #             results_folder = os.path.join('media', 'results')
# # #             os.makedirs(results_folder, exist_ok=True)

# # #             # Generate the annotated image
# # #             annotated_image_path = os.path.join(results_folder, f"annotated_{os.path.basename(image_path)}")
# # #             annotated_image = results[0].plot()  # Generate the annotated image
# # #             cv2.imwrite(annotated_image_path, annotated_image)  # Save the annotated image

# # #             # Extract detected labels, confidence scores, bounding boxes, and calculate area from masks
# # #             detected_objects = []
# # #             for i, (box, cls, conf, mask) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf, results[0].masks)):
# # #                 # Convert tensor to list for bounding box
# # #                 box = box.tolist()
# # #                 x1, y1, x2, y2 = box

# # #                 # Extract the segmentation (xy) for the mask if available
# # #                 if mask.xy:  # Ensure mask has valid segmentation data
# # #                     mask_xy = mask.xy  # Segmentation data in pixel coordinates
                    
# # #                     # Calculate the mask area as the number of segments
# # #                     mask_area = sum([len(segment) for segment in mask_xy])  # Count the number of pixels in the mask

# # #                     detected_objects.append({
# # #                         "label": results[0].names[int(cls)],
# # #                         "confidence": float(conf.item()),  # Convert tensor to float
# # #                         "bounding_box": box,  # Bounding box as list
# # #                         "area": f"{mask_area}px",  # Area from the mask in pixels
# # #                     })
# # #                 else:
# # #                     detected_objects.append({
# # #                         "label": results[0].names[int(cls)],
# # #                         "confidence": float(conf.item()),  # Convert tensor to float
# # #                         "bounding_box": box,  # Bounding box as list
# # #                         "area": "0px",  # No area available if no mask
# # #                     })

# # #             # Return a response with the annotated image path and detected objects
# # #             return JsonResponse({
# # #                 'status': 'success',
# # #                 'image_path': annotated_image_path,
# # #                 'detected_objects': detected_objects
# # #             })

# # #         except Exception as e:
# # #             return JsonResponse({
# # #                 'status': 'error',
# # #                 'message': str(e)
# # #             }, status=500)


# # # import os
# # # import cv2
# # # from rest_framework.views import APIView
# # # from rest_framework.parsers import MultiPartParser, FormParser
# # # from rest_framework.response import Response
# # # from django.http import JsonResponse
# # # from .models import UploadedImage
# # # from ultralytics import YOLO

# # # class ImageUploadView(APIView):
# # #     parser_classes = (MultiPartParser, FormParser)

# # #     def post(self, request, *args, **kwargs):
# # #         try:
# # #             # Save the uploaded image
# # #             uploaded_image = request.FILES['image']
# # #             image_instance = UploadedImage.objects.create(image=uploaded_image)

# # #             # Load the YOLO model
# # #             model = YOLO('app/trainingmodel/best.pt')

# # #             # Perform inference on the uploaded image
# # #             image_path = image_instance.image.path
# # #             results = model(image_path)

# # #             # Create a folder for saving annotated images
# # #             results_folder = os.path.join('media', 'results')
# # #             os.makedirs(results_folder, exist_ok=True)

# # #             # Generate the annotated image
# # #             annotated_image_path = os.path.join(results_folder, f"annotated_{os.path.basename(image_path)}")
# # #             annotated_image = results[0].plot()  # Generate the annotated image
# # #             cv2.imwrite(annotated_image_path, annotated_image)  # Save the annotated image

# # #             # Extract detected labels, confidence scores, bounding boxes, and calculate area from masks
# # #             detected_objects = []
# # #             for i, (box, cls, conf, mask) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf, results[0].masks)):
# # #                 # Convert tensor to list for bounding box
# # #                 box = box.tolist()
# # #                 x1, y1, x2, y2 = box

# # #                 # Extract the segmentation (xy) for the mask if available
# # #                 if mask.xy:  # Ensure mask has valid segmentation data
# # #                     mask_xy = mask.xy  # Segmentation data in pixel coordinates
                    
# # #                     # Calculate the mask area as the number of segments
# # #                     mask_area = sum([len(segment) for segment in mask_xy])  # Count the number of pixels in the mask

# # #                     detected_objects.append({
# # #                         "label": results[0].names[int(cls)],
# # #                         "confidence": float(conf.item()),  # Convert tensor to float
# # #                         "bounding_box": box,  # Bounding box as list
# # #                         "area": f"{mask_area}px",  # Area from the mask in pixels
# # #                     })
# # #                 else:
# # #                     detected_objects.append({
# # #                         "label": results[0].names[int(cls)],
# # #                         "confidence": float(conf.item()),  # Convert tensor to float
# # #                         "bounding_box": box,  # Bounding box as list
# # #                         "area": "0px",  # No area available if no mask
# # #                     })

# # #             # Return a response with the annotated image path and detected objects
# # #             return JsonResponse({
# # #                 'status': 'success',
# # #                 'image_path': annotated_image_path,
# # #                 'detected_objects': detected_objects
# # #             })

# # #         except Exception as e:
# # #             return JsonResponse({
# # #                 'status': 'error',
# # #                 'message': str(e)
# # #             }, status=500)




# # import os
# # import cv2
# # import numpy as np
# # from rest_framework.views import APIView
# # from rest_framework.parsers import MultiPartParser, FormParser
# # from rest_framework.response import Response
# # from django.http import JsonResponse
# # from .models import UploadedImage
# # from ultralytics import YOLO

# # class ImageUploadView(APIView):
# #     parser_classes = (MultiPartParser, FormParser)

# #     def post(self, request, *args, **kwargs):
# #         try:
# #             # Save the uploaded image
# #             uploaded_image = request.FILES['image']
# #             image_instance = UploadedImage.objects.create(image=uploaded_image)

# #             # Load the YOLO model
# #             model = YOLO('app/trainingmodel/best.pt')

# #             # Perform inference on the uploaded image
# #             image_path = image_instance.image.path
# #             results = model(image_path)

# #             # Create a folder for saving annotated images
# #             results_folder = os.path.join('media', 'results')
# #             os.makedirs(results_folder, exist_ok=True)

# #             # Generate the annotated image
# #             annotated_image_path = os.path.join(results_folder, f"annotated_{os.path.basename(image_path)}")
# #             annotated_image = results[0].plot()  # Generate the annotated image
# #             cv2.imwrite(annotated_image_path, annotated_image)  # Save the annotated image

# #             # Extract detected labels, confidence scores, bounding boxes, and calculate area from masks
# #             detected_objects = []
# #             for i, (box, cls, conf, mask) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf, results[0].masks)):
# #                 # Convert tensor to list for bounding box
# #                 box = box.tolist()
# #                 x1, y1, x2, y2 = box

# #                 # Extract the segmentation (xy) for the mask if available
# #                 if mask.xy:  # Ensure mask has valid segmentation data
# #                     mask_xy = mask.xy  # Segmentation data in pixel coordinates
                    
# #                     # Convert the mask segmentation into a binary mask image
# #                     mask_img = np.zeros((image_instance.image.height, image_instance.image.width), dtype=np.uint8)
# #                     for polygon in mask_xy:
# #                         pts = np.array(polygon, dtype=np.int32)
# #                         pts = pts.reshape((-1, 1, 2))  # Reshape to the format needed by OpenCV
# #                         cv2.fillPoly(mask_img, [pts], 255)  # Fill the polygon in the mask image

# #                     # Calculate the area of the mask
# #                     mask_area = cv2.countNonZero(mask_img)  # Count the number of non-zero pixels in the mask image

# #                     detected_objects.append({
# #                         "label": results[0].names[int(cls)],
# #                         "confidence": float(conf.item()),  # Convert tensor to float
# #                         "bounding_box": box,  # Bounding box as list
# #                         "area": f"{mask_area}px",  # Area from the mask in pixels
# #                     })
# #                 else:
# #                     detected_objects.append({
# #                         "label": results[0].names[int(cls)],
# #                         "confidence": float(conf.item()),  # Convert tensor to float
# #                         "bounding_box": box,  # Bounding box as list
# #                         "area": "0px",  # No area available if no mask
# #                     })

# #             # Return a response with the annotated image path and detected objects
# #             return JsonResponse({
# #                 'status': 'success',
# #                 'image_path': annotated_image_path,
# #                 'detected_objects': detected_objects
# #             })

# #         except Exception as e:
# #             return JsonResponse({
# #                 'status': 'error',
# #                 'message': str(e)
# #             }, status=500)



# import os
# import cv2
# import numpy as np
# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser, FormParser
# from django.http import JsonResponse
# from .models import UploadedImage
# from ultralytics import YOLO

# class ImageUploadView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request, *args, **kwargs):
#         try:
#             # Save the uploaded image
#             uploaded_image = request.FILES['image']
#             image_instance = UploadedImage.objects.create(image=uploaded_image)

#             # Load the YOLO model
#             model = YOLO('app/trainingmodel/best.pt')

#             # Perform inference on the uploaded image
#             image_path = image_instance.image.path
#             results = model(image_path)

#             # Create a folder for saving annotated images
#             results_folder = os.path.join('media', 'results')
#             os.makedirs(results_folder, exist_ok=True)

#             # Generate the annotated image
#             annotated_image_path = os.path.join(results_folder, f"annotated_{os.path.basename(image_path)}")
#             annotated_image = results[0].plot()  # Generate the annotated image
#             cv2.imwrite(annotated_image_path, annotated_image)  # Save the annotated image

#             # Initialize detected data for the response
#             detected_data = []

#             for i, (box, cls, conf, mask) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf, results[0].masks)):
#                 # Extract mask information if available
#                 tooth_area = "None"
#                 pulp_area = "None"

#                 if mask.xy:  # Ensure mask has valid segmentation data
#                     mask_xy = mask.xy  # Segmentation data in pixel coordinates
#                     mask_img = np.zeros((image_instance.image.height, image_instance.image.width), dtype=np.uint8)

#                     for polygon in mask_xy:
#                         pts = np.array(polygon, dtype=np.int32)
#                         pts = pts.reshape((-1, 1, 2))  # Reshape to the format needed by OpenCV
#                         cv2.fillPoly(mask_img, [pts], 255)  # Fill the polygon in the mask image

#                     # Calculate areas of interest
#                     mask_area = cv2.countNonZero(mask_img)
#                     tooth_area = f"{mask_area}px"  # Assign mask area as tooth area

#                     # Check class type and assign pulp area if applicable
#                     if results[0].names[int(cls)] == "pulp":
#                         pulp_area = f"{mask_area}px"
#                     else:
#                         pulp_area = "None"

#                 # Append data for each detection
#                 detected_data.append({
#                     "tooth_area_in_pixels": tooth_area,
#                     "pulp_area_in_pixels": pulp_area
#                 })

#             # Construct response
#             response_data = {
#                 "StatusCode": 200,
#                 "Image Path": f"http://127.0.0.1:8001/static/results/{os.path.basename(annotated_image_path)}",
#                 "Data": detected_data
#             }

#             return JsonResponse(response_data)

#         except Exception as e:
#             return JsonResponse({
#                 "StatusCode": 500,
#                 "Message": str(e)
#             }, status=500)

# Working Calculate the Area 

# import os
# import cv2
# import numpy as np
# from rest_framework.views import APIView
# from rest_framework.parsers import MultiPartParser, FormParser
# from django.http import JsonResponse
# from .models import UploadedImage
# from ultralytics import YOLO

# class ImageUploadView(APIView):
#     parser_classes = (MultiPartParser, FormParser)

#     def post(self, request, *args, **kwargs):
#         try:
#             # Save the uploaded image
#             uploaded_image = request.FILES['image']
#             image_instance = UploadedImage.objects.create(image=uploaded_image)
#             print(uploaded_image)
#             # Load the YOLO model
#             model = YOLO('app/trainingmodel/model.pt')
#             print(model)
#             # Perform inference on the uploaded image
#             image_path = image_instance.image.path
#             results = model(image_path)
#             print(results)
#             # Create a folder for saving annotated images
#             results_folder = os.path.join('media', 'results')
#             os.makedirs(results_folder, exist_ok=True)

#             # Generate the annotated image
#             annotated_image_path = os.path.join(results_folder, f"annotated_{os.path.basename(image_path)}")
#             annotated_image = results[0].plot()  # Generate the annotated image
#             cv2.imwrite(annotated_image_path, annotated_image)  # Save the annotated image

#             # Group detections by proximity to associate `tooth` and `pulp` areas
#             detected_data = []
#             detection_map = {}  # Map to associate tooth and pulp areas

#             for i, (box, cls, conf, mask) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf, results[0].masks)):
#                 # Class name
#                 class_name = results[0].names[int(cls)]

#                 # Calculate the mask area if available
#                 area = "None"
#                 if mask.xy:
#                     mask_img = np.zeros((image_instance.image.height, image_instance.image.width), dtype=np.uint8)
#                     for polygon in mask.xy:
#                         pts = np.array(polygon, dtype=np.int32)
#                         pts = pts.reshape((-1, 1, 2))  # Reshape for OpenCV
#                         cv2.fillPoly(mask_img, [pts], 255)  # Fill the polygon
#                     area = str(cv2.countNonZero(mask_img))  # Non-zero pixel count

#                 # Group data by proximity or same object detection
#                 if class_name == "tooth":
#                     detection_map[i] = {"tooth_area_in_pixels": area, "pulp_area_in_pixels": "None"}
#                 elif class_name == "pulp":
#                     # Find the nearest `tooth` detection to associate the `pulp` area
#                     closest_key = min(detection_map.keys(), key=lambda k: abs(box[0] - results[0].boxes.xyxy[k][0]))
#                     detection_map[closest_key]["pulp_area_in_pixels"] = area

#             # Convert detection_map to a list
#             detected_data = list(detection_map.values())

#             # Construct response
#             response_data = {
#                 "StatusCode": 200,
#                 "Image Path": f"http://127.0.0.1:8001/static/results/{os.path.basename(annotated_image_path)}",
#                 "Data": detected_data
#             }

#             return JsonResponse(response_data)

#         except Exception as e:
#             return JsonResponse({
#                 "StatusCode": 500,
#                 "Message": str(e)
#             }, status=500)



# testing Area 



import os
import cv2
import numpy as np
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse
from .models import UploadedImage
from ultralytics import YOLO

class ImageUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            # 1. Save the uploaded image
            uploaded_image = request.FILES['image']
            image_instance = UploadedImage.objects.create(image=uploaded_image)

            # 2. Load the YOLO model
            model = YOLO('app/trainingmodel/model.pt')  # Path to YOLO model

            # 3. Perform inference on the uploaded image
            image_path = image_instance.image.path
            results = model(image_path , conf=0.25)

            # 4. Create folder for saving annotated images
            results_folder = os.path.join('media', 'results')
            os.makedirs(results_folder, exist_ok=True)

            # Save annotated image
            annotated_image_path = os.path.join(results_folder, f"annotated_{os.path.basename(image_path)}")
            annotated_image = results[0].plot()
            cv2.imwrite(annotated_image_path, annotated_image)

            # 5. Initialize Detection Map for grouping
            detection_map = {}  # Maps tooth index -> tooth and pulp area details

            # 6. Function to calculate mask area
            def calculate_mask_area(mask, img_height, img_width):
                mask_img = np.zeros((img_height, img_width), dtype=np.uint8)
                for polygon in mask.xy:  # For each polygon in the mask
                    pts = np.array(polygon, dtype=np.int32)
                    pts = pts.reshape((-1, 1, 2))  # Reshape for OpenCV fill
                    cv2.fillPoly(mask_img, [pts], 255)  # Fill the polygon
                return cv2.countNonZero(mask_img)  # Return area as pixel count

            # 7. Process detections: Loop through results and calculate areas
            image_height, image_width = cv2.imread(image_path).shape[:2]
            for i, (box, cls, mask) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].masks)):
                class_name = results[0].names[int(cls)]

                if mask is not None:
                    area = calculate_mask_area(mask, image_height, image_width)
                else:
                    area = "None"

                # Group data: Initialize tooth areas, associate pulp to the nearest tooth
                if class_name == "tooth":
                    detection_map[i] = {"tooth_area_in_pixels": area, "pulp_area_in_pixels": "None"}
                elif class_name == "pulp":
                    # Find nearest tooth detection
                    nearest_tooth_index = min(
                        detection_map.keys(),
                        key=lambda k: abs(box[0] - results[0].boxes.xyxy[k][0])
                    )
                    detection_map[nearest_tooth_index]["pulp_area_in_pixels"] = area

            # 8. Format the response data
            detected_data = list(detection_map.values())

            response_data = {
                "StatusCode": 200,
                "Image Path": f"http://127.0.0.1:8001/static/results/{os.path.basename(annotated_image_path)}",
                "Data": detected_data
            }

            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({
                "StatusCode": 500,
                "Message": str(e)
            }, status=500)


