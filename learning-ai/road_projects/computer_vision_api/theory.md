# 👁️ Computer Vision API - Lý thuyết

> **Mục tiêu**: Xây dựng API xử lý hình ảnh với các task: classification, detection, segmentation, OCR

## 🧠 **Lý thuyết cơ bản**

### **1. Computer Vision Tasks**

**Core Tasks:**
- **Image Classification**: Phân loại hình ảnh vào các categories
- **Object Detection**: Phát hiện và định vị objects trong hình ảnh
- **Image Segmentation**: Phân đoạn hình ảnh theo pixels
- **Optical Character Recognition (OCR)**: Nhận dạng text trong hình ảnh
- **Face Recognition**: Nhận dạng và xác thực khuôn mặt
- **Image Generation**: Tạo hình ảnh từ text hoặc sketches

### **2. Deep Learning Architectures**

**A. Convolutional Neural Networks (CNNs):**
- **LeNet-5**: Architecture đầu tiên cho digit recognition
- **AlexNet**: Breakthrough trong ImageNet 2012
- **VGG**: Deep networks với 3x3 convolutions
- **ResNet**: Residual connections để train deep networks
- **EfficientNet**: Compound scaling method

**B. Object Detection Models:**
- **R-CNN Family**: Region-based approaches
- **YOLO**: Real-time object detection
- **SSD**: Single Shot Detector
- **RetinaNet**: Focal loss for dense detection

**C. Segmentation Models:**
- **U-Net**: Medical image segmentation
- **DeepLab**: Atrous convolutions
- **Mask R-CNN**: Instance segmentation
- **SegNet**: Encoder-decoder architecture

### **3. Model Optimization Techniques**

**A. Model Compression:**
- **Quantization**: Reduce precision (FP32 → INT8)
- **Pruning**: Remove unnecessary weights
- **Knowledge Distillation**: Transfer knowledge từ teacher sang student
- **Model Architecture Search (NAS)**: AutoML cho architecture design

**B. Inference Optimization:**
- **TensorRT**: NVIDIA's inference optimizer
- **ONNX**: Open Neural Network Exchange
- **TorchScript**: PyTorch model serialization
- **TensorFlow Lite**: Mobile/edge deployment

## 🔧 **Technical Architecture**

### **1. Computer Vision API Architecture**

```python
class ComputerVisionArchitecture:
    """Architecture cho Computer Vision API"""
    
    def __init__(self):
        self.components = {
            'data_processing': ['Image Preprocessing', 'Augmentation', 'Normalization'],
            'model_serving': ['Model Loading', 'Inference Engine', 'Batch Processing'],
            'post_processing': ['NMS', 'Thresholding', 'Format Conversion'],
            'api_layer': ['REST API', 'WebSocket', 'gRPC'],
            'monitoring': ['Performance Metrics', 'Error Tracking', 'Model Drift']
        }
    
    def explain_data_flow(self):
        """Explain data flow trong hệ thống"""
        print("""
        **Computer Vision API Data Flow:**
        
        1. **Input Processing Layer:**
           - Image upload và validation
           - Format conversion (JPEG, PNG, WebP)
           - Resize và normalization
           - Batch preparation
        
        2. **Model Inference Layer:**
           - Model loading và warmup
           - GPU/CPU inference
           - Batch processing optimization
           - Memory management
        
        3. **Post-Processing Layer:**
           - Non-maximum suppression (NMS)
           - Confidence thresholding
           - Result formatting và validation
           - Error handling
        
        4. **API Response Layer:**
           - JSON response formatting
           - Image annotation overlay
           - Metadata extraction
           - Caching strategies
        
        5. **Monitoring Layer:**
           - Inference latency tracking
           - Model accuracy monitoring
           - Resource utilization
           - Error rate analysis
        """)
```

### **2. Image Classification Implementation**

**ResNet Classification:**
```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageClassifier:
    """Image Classification using ResNet"""
    
    def __init__(self, model_name='resnet50', num_classes=1000, pretrained=True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify final layer if needed
        if num_classes != 1000:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load ImageNet classes
        self.class_names = self._load_imagenet_classes()
    
    def _load_imagenet_classes(self):
        """Load ImageNet class names"""
        # In practice, load from file
        return [f"class_{i}" for i in range(1000)]
    
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor.unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    def predict(self, image_path, top_k=5):
        """Predict image classification"""
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_path)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Format results
            predictions = []
            for i in range(top_k):
                class_idx = top_indices[0][i].item()
                confidence = top_probs[0][i].item()
                class_name = self.class_names[class_idx]
                
                predictions.append({
                    'class_id': class_idx,
                    'class_name': class_name,
                    'confidence': confidence
                })
            
            return {
                'predictions': predictions,
                'model_name': self.model_name,
                'inference_time': None  # Add timing if needed
            }
            
        except Exception as e:
            raise RuntimeError(f"Prediction error: {str(e)}")
    
    def batch_predict(self, image_paths, batch_size=32):
        """Batch prediction for multiple images"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            # Preprocess batch
            for path in batch_paths:
                tensor = self.preprocess_image(path)
                batch_tensors.append(tensor)
            
            # Stack tensors
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            
            # Process results
            for j, path in enumerate(batch_paths):
                probs = probabilities[j]
                top_probs, top_indices = torch.topk(probs, 5)
                
                predictions = []
                for k in range(5):
                    class_idx = top_indices[k].item()
                    confidence = top_probs[k].item()
                    class_name = self.class_names[class_idx]
                    
                    predictions.append({
                        'class_id': class_idx,
                        'class_name': class_name,
                        'confidence': confidence
                    })
                
                results.append({
                    'image_path': path,
                    'predictions': predictions
                })
        
        return results
```

### **3. Object Detection Implementation**

**YOLO Detection:**
```python
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class ObjectDetector:
    """Object Detection using YOLO"""
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5, iou_threshold=0.45):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # COCO class names
        self.class_names = self._load_coco_classes()
    
    def _load_coco_classes(self):
        """Load COCO class names"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def detect_objects(self, image_path):
        """Detect objects in image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id]
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        detections.append(detection)
            
            return {
                'detections': detections,
                'image_path': image_path,
                'model_name': self.model_path,
                'total_objects': len(detections)
            }
            
        except Exception as e:
            raise RuntimeError(f"Detection error: {str(e)}")
    
    def draw_detections(self, image_path, output_path=None):
        """Draw detections on image"""
        try:
            # Detect objects
            result = self.detect_objects(image_path)
            
            # Load image
            image = cv2.imread(image_path)
            
            # Draw bounding boxes
            for detection in result['detections']:
                bbox = detection['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save or return image
            if output_path:
                cv2.imwrite(output_path, image)
                return output_path
            else:
                return image
                
        except Exception as e:
            raise RuntimeError(f"Drawing error: {str(e)}")
```

### **4. OCR Implementation**

**Tesseract OCR:**
```python
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re

class OCRProcessor:
    """Optical Character Recognition using Tesseract"""
    
    def __init__(self, tesseract_path=None, language='eng'):
        self.language = language
        
        # Set tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # OCR configuration
        self.config = '--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Dilation
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        return dilated
    
    def extract_text(self, image_path, preprocess=True):
        """Extract text from image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # Preprocess if needed
            if preprocess:
                processed_image = self.preprocess_image(image)
            else:
                processed_image = image
            
            # Extract text
            text = pytesseract.image_to_string(
                processed_image,
                lang=self.language,
                config=self.config
            )
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            return {
                'text': cleaned_text,
                'raw_text': text,
                'image_path': image_path,
                'language': self.language,
                'word_count': len(cleaned_text.split())
            }
            
        except Exception as e:
            raise RuntimeError(f"OCR error: {str(e)}")
    
    def extract_text_with_boxes(self, image_path, preprocess=True):
        """Extract text with bounding boxes"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # Preprocess if needed
            if preprocess:
                processed_image = self.preprocess_image(image)
            else:
                processed_image = image
            
            # Extract text with boxes
            data = pytesseract.image_to_data(
                processed_image,
                lang=self.language,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            # Process results
            text_boxes = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Filter by confidence
                    text_box = {
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'bbox': [
                            data['left'][i],
                            data['top'][i],
                            data['left'][i] + data['width'][i],
                            data['top'][i] + data['height'][i]
                        ]
                    }
                    text_boxes.append(text_box)
            
            return {
                'text_boxes': text_boxes,
                'image_path': image_path,
                'language': self.language,
                'total_words': len(text_boxes)
            }
            
        except Exception as e:
            raise RuntimeError(f"OCR with boxes error: {str(e)}")
    
    def _clean_text(self, text):
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_specific_info(self, image_path, info_type='email'):
        """Extract specific information from image"""
        try:
            # Extract text
            result = self.extract_text(image_path)
            text = result['text']
            
            # Define patterns for different info types
            patterns = {
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
                'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                'price': r'\$\d+(?:\.\d{2})?',
                'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
            }
            
            if info_type not in patterns:
                raise ValueError(f"Unsupported info type: {info_type}")
            
            # Find matches
            matches = re.findall(patterns[info_type], text)
            
            return {
                'info_type': info_type,
                'matches': matches,
                'text': text,
                'image_path': image_path
            }
            
        except Exception as e:
            raise RuntimeError(f"Info extraction error: {str(e)}")
```

## 📊 **Performance Optimization**

### **1. Model Optimization**

**Quantization Example:**
```python
import torch
import torch.quantization as quantization

class OptimizedModel:
    """Model optimization for deployment"""
    
    def __init__(self, model):
        self.model = model
        self.optimized_model = None
    
    def quantize_model(self, calibration_data):
        """Quantize model to INT8"""
        # Set model to evaluation mode
        self.model.eval()
        
        # Prepare for quantization
        self.model.qconfig = quantization.get_default_qconfig('fbgemm')
        
        # Prepare calibration
        quantization.prepare(self.model, inplace=True)
        
        # Calibrate with data
        with torch.no_grad():
            for data in calibration_data:
                self.model(data)
        
        # Convert to quantized model
        self.optimized_model = quantization.convert(self.model, inplace=False)
        
        return self.optimized_model
    
    def save_optimized_model(self, path):
        """Save optimized model"""
        if self.optimized_model:
            torch.save(self.optimized_model.state_dict(), path)
    
    def load_optimized_model(self, path):
        """Load optimized model"""
        self.optimized_model.load_state_dict(torch.load(path))
        self.optimized_model.eval()
```

### **2. Batch Processing**

**Efficient Batch Processing:**
```python
class BatchProcessor:
    """Efficient batch processing for computer vision"""
    
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
    
    def process_batch(self, images):
        """Process batch of images efficiently"""
        # Preprocess batch
        batch_tensors = []
        for image in images:
            tensor = self.preprocess_image(image)
            batch_tensors.append(tensor)
        
        # Stack tensors
        batch = torch.stack(batch_tensors).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(batch)
        
        # Post-process results
        results = self.postprocess_batch(outputs)
        
        return results
    
    def preprocess_image(self, image):
        """Preprocess single image"""
        # Implementation depends on model requirements
        pass
    
    def postprocess_batch(self, outputs):
        """Post-process batch outputs"""
        # Implementation depends on model outputs
        pass
```

## 🎯 **Business Impact**

### **Expected Outcomes:**
- **Automated Processing**: Reduce manual image analysis by 80%
- **Real-time Analysis**: Process images in milliseconds
- **Scalable Solution**: Handle thousands of images per hour
- **Accuracy Improvement**: 95%+ accuracy for most tasks
- **Cost Reduction**: 60% reduction in image processing costs

---

**📚 References:**
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Computer Vision: Algorithms and Applications" by Richard Szeliski
- "YOLO: Real-Time Object Detection" by Redmon et al.
- "ResNet: Deep Residual Learning for Image Recognition" by He et al.