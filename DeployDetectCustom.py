# -*- coding: utf-8 -*-

import os
import cv2
import torch 
import PySpin
import datetime
import tempfile
import torch.backends.cudnn as cudnn 

#from numpy import random 
from pathlib import Path

from models.experimental import attempt_load 
from utils.datasets import LoadStreams, LoadImages 
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device 


system = PySpin.System.GetInstance()


# Get the camera list 
cam_list = system.GetCameras()

num_cameras = cam_list.GetSize()
if num_cameras == 0:
    raise Exception("No cameras found!")
    
    
# Select the first camera 
camera = cam_list.GetByIndex(0)

# Initiate camera 
camera.Init()


# Create directories
output_directory1 = "expRun/captured_images"
if not os.path.exists(output_directory1):
    os.makedirs(output_directory1)

output_directory2 = "expRun/detected_images"
if not os.path.exists(output_directory2):
    os.makedirs(output_directory2)


# Acquisition control parameters
camera.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
camera.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
camera.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
camera.ExposureTime.SetValue(800)
camera.GainAuto.SetValue(PySpin.GainAuto_Off)
camera.Gain.SetValue(8.5)
camera.GammaEnable.SetValue(True)
camera.Gamma.SetValue(0.8)
camera.BalanceWhiteAuto.SetValue(PySpin.BalanceWhiteAuto_Off)
camera.BlackLevelSelector.SetValue(PySpin.BlackLevelSelector_All)
camera.BlackLevel.SetValue(1.7)
camera.BalanceRatioSelector.SetValue(PySpin.BalanceRatioSelector_Red)
camera.BalanceRatio.SetValue(1.74)


# Trigger sensor settings
camera.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
camera.TriggerMode.SetValue(PySpin.TriggerMode_On)
camera.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
camera.TriggerActivation.SetValue(PySpin.TriggerActivation_FallingEdge)
camera.TriggerOverlap.SetValue(PySpin.TriggerOverlap_Off)
camera.TriggerDelay.SetValue(68)


# Begin acquiring images 
camera.BeginAcquisition()
print("Camera started acquiring images.")


def detect(weights, model, source, conf_thres, iou_thres, img_size, view_img, save_img, save_txt):
    save_img = not save_img and source.endswith('.txt') 
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
        
    # Set dataloader
    #vid_path, vid_writer = None, None 
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True # Set True to speed up constant image size inference 
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)     # PROJECT LINE EXECUTE.!


    # Directories 
    save_dir = Path('expRun/detected_images') # Output directory 
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) # Make dir 

    
    # Get names and colors 
    names = model.module.names if hasattr(model, 'module') else model.names 
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    
    # Run inference 
    #if device.type != 'cpu':
    #    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    old_img_w = old_img_h = imgsz 
    old_img_b = 1 
    
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() # uint8 to fp16/32 
        img /= 255.0 # 0- 255 to 0.0 - 1.0 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
            
        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]
            
            
        # Inference 
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        
        
        # Apply NMS 
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic= False)
    
        output_file = open("label.txt", "a")    
    
    
        # Process detections 
        for i, det in enumerate(pred): # Detection
            if webcam: # batch_size >= 1
                p, s, im0, frame = path[i], '%g:' % i, im0s[i].copy(), dataset.count
            else: 
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)  # PROJECT LINE EXECUTE.!
            
            
            p = Path(p) # To Path 
            #save_path = str(save_dir / p.name) # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}') # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] # Nomalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size 
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                
                # print results 
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum() # Detection per class 
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, " # Add to string 
                
                      
                # Write results 
                for *xyxy, conf, cls in reversed(det):
                    if save_txt: # Write to file 
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() # Normalized xywh
                        line = (cls, *xywh, conf) if False else (cls, *xywh) # Label format 
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                          
                            
                    if save_img or view_img: # Add bbox to image 
                        label = f'{names[int(cls)]} {conf:.2f}'
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        plot_one_box(xyxy, im0, label=label, color=(0, 0, 128), line_thickness=2)
                        print(label)
                        
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        output_file.write(f'{timestamp} - {label}\n')
                        
                        image_file = f"detected_image_{image_count}.jpg"
                        image_file = os.path.join(output_directory2, image_file)
                        cv2.imwrite(image_file, im0)
                        
        output_file.close()
                

weights = 'yolov7.pt'
conf_thres = 0.3
iou_thres = 0.7 
img_size = 640
view_img = True
save_img = True
save_txt = False

set_logging()
device = select_device('')
half = device.type != 'cpu' # Half precision only supported on CUDA
model = attempt_load(weights, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(img_size, s=stride)
if half:
    model.half()
model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))



try: 
    image_count = 0
    captured_image_path = None 
    
    # Wait for trigger signal to activate capture
    while True: 
        # Wait for the trigger signal (e.g., GPIO input) to indicate image capture
        # Once the trigger signal is received, procees with image capture
        
        # Retrieve the next image from the camera 
        image_result = camera.GetNextImage()
        
        if image_result.IsIncomplete():
            print("Image incomplete with status %d." % image_result.GetImageStatus())
        else:
            # Convert the image result to a numpy array 
            image_data = image_result.GetNDArray()
            
            colored_image = cv2.cvtColor(image_data, cv2.COLOR_BAYER_RG2RGB)
            
            # Resize 
            captured_image = cv2.resize(colored_image, (2048, 1080))
            
            # Crop image using ROI coordinates 
            cropped_image = captured_image[230: 930, 400: 1800]
            
            
            # Save the image as a temporary file 
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                captured_image_path = temp_file.name 
                cv2.imwrite(captured_image_path, cropped_image)
                
                
            # Release the image 
            image_result.Release()
            
            
            # Save image locally
            image_filename = f"captured_image_{image_count}.jpg"
            image_filename = os.path.join(output_directory1, image_filename)
            cv2.imwrite(image_filename, cropped_image)
            image_count += 1 
            
            
            if captured_image_path is not None:
                source = captured_image_path
                detect(weights, model, source, conf_thres, iou_thres, img_size, view_img, save_img, save_txt)
                
        
        # Check for a key press event
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'): # Exit loop if 'q' key is pressed 
            break 
        
    
except KeyboardInterrupt:
    # Stop image acquisition 
    camera.EndAcquisition()
    print("Image acquisition stopped.")
    
    # Deinitialize the camera
    camera.Deinit()
    print("Camera deninitialized.")
    
    
# Release the camera and system resources 
del camera 
cam_list.Clear()
system.ReleaseInstances()
cv2.destroyAllWindows()
            
