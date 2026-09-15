/**
 * Built-in Model Catalog for Standalone Edge Inference.
 * Contains metadata, class definitions, supported resolutions, and public GitHub CDN endpoints.
 * Completely standalone — downloads direct from high-speed GitHub Raw CDN into browser CacheStorage.
 */

export const YOLO_CLASSES_27 = [
  "Bike", "Building", "Car", "Person", "Stairs", "Traffic sign",
  "Electrical Pole", "Road", "Motorcycle", "Dustbin", "Dog", "Manhole",
  "Tree", "Guard rail", "Pedestrian crosswalk", "Truck", "Bus", "Bench",
  "Traffic Cone", "Fire hydrant", "Teraffic Barrel", "Plant Pot",
  "Electrical Box", "Chair", "Bicycle Rack", "Door", "Wall"
];

export const YOLO_CLASSES_25 = [
  "Bike/Motorcycle", "Building", "Vehicle", "Person", "Stairs",
  "Traffic sign", "Electrical Pole", "Dustbin", "Animal", "Manhole",
  "Tree", "Guard rail", "Pedestrian crosswalk", "Bench", "Traffic Cone",
  "Teraffic Barrel", "Plant Pot", "Chair", "Door", "Table/Desk",
  "Bookshelf/Storage", "Window", "Sign_Board", "Display", "Drawer"
];

export const YOLO_MODELS = [
  {
    id: "yolo26n_nav_run8",
    name: "⚡ YOLO26-Nano Fast (Run 8 - 256p - 25 Classes)",
    variant: "nano",
    classes: 25,
    classNames: YOLO_CLASSES_25,
    supportedResolutions: [256],
    defaultResolution: 256,
    approxSizeMb: 9.2,
    cdnUrls: {
      256: "https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26n_nav_run8_256.onnx"
    },
    inputTensorName: "images",
    outputTensorName: "output0"
  },
  {
    id: "yolo26s_nav_run683",
    name: "⚡ YOLO26-Small Balanced (Run 683 - 416p - 25 Classes)",
    variant: "small",
    classes: 25,
    classNames: YOLO_CLASSES_25,
    supportedResolutions: [416],
    defaultResolution: 416,
    approxSizeMb: 36.3,
    cdnUrls: {
      416: "https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/yolo26s_nav_run683_416.onnx"
    },
    inputTensorName: "images",
    outputTensorName: "output0"
  }
];

export const DEPTH_MODELS = [
  {
    id: "yolo26_depth",
    name: "📐 YOLO26-Nano Depth Metric (512 Quality / 320 Fast / 256 Low-Power)",
    variant: "yolo_depth",
    isMetric: true,
    supportedResolutions: [512, 320, 256],
    defaultResolution: 512,
    approxSizeMb: 19.8,
    cdnUrls: {
      256: "https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/depth_256.onnx",
      320: "https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/depth_320.onnx",
      512: "https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/depth_512.onnx"
    },
    inputTensorName: "images",
    outputTensorName: "output0"
  },
  {
    id: "yolo26s_depth",
    name: "📐 YOLO26-Small Depth Metric (320 Fast / 256 Low-Power)",
    variant: "yolo_depth",
    isMetric: true,
    supportedResolutions: [320, 256],
    defaultResolution: 320,
    approxSizeMb: 46.0,
    cdnUrls: {
      256: "https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/yolo26s_depth_256.onnx",
      320: "https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/yolo26s_depth_320.onnx"
    },
    inputTensorName: "images",
    outputTensorName: "output0"
  },
  {
    id: "yolo26m_depth",
    name: "📐 YOLO26-Medium Depth Metric (320 Accurate / 256 Low-Power)",
    variant: "yolo_depth",
    isMetric: true,
    supportedResolutions: [320, 256],
    defaultResolution: 256,
    approxSizeMb: 84.3,
    cdnUrls: {
      256: "https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/yolo26m_depth_256.onnx",
      320: "https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth/yolo26m_depth_320.onnx"
    },
    inputTensorName: "images",
    outputTensorName: "output0"
  },
  {
    id: "depth_anything_v2_small_quantized",
    name: "🗺️ Depth Anything V2 Small (Quantized 518p)",
    variant: "depth_anything",
    isMetric: false,
    supportedResolutions: [518],
    defaultResolution: 518,
    approxSizeMb: 26.0,
    cdnUrls: {
      518: "https://raw.githubusercontent.com/Ashutosh-Ranjan310106/yolo-model-edge-inference/main/models/depth_anything_v2_small_quantized.onnx"
    },
    inputTensorName: "pixel_values",
    outputTensorName: "predicted_depth"
  }
];
