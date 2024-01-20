# Production Challenge
Pipeline example for optimization and deployment of an object detector model.

Table of contents
-----------------
* [Model Comparison](#comparison)    
* [Optimization](#optimization)
* [Deployment](#deployment)

<a name="comparison"></a>
Model Comparison
------------
For the first step I will be comparing two models from the yolo series. For a fair comparison, both models come from a familiar library, Ultralytics. By choosing a similar framework the statistics and metrics will be fair, also taking into consideration the parameters and input size.

- Models:
    - [Yolov8](https://github.com/ultralytics/ultralytics)
    - [Yolov5](https://github.com/ultralytics/yolov5)
- Size (Parameters):
    - Yolov8 Nano: 3.2 M
    - Yolov5 Nano: 1.9 M
- FLOPs
    - Yolov8 Nano: 8.7 B
    - Yolov5 Nano: 4.5 B
- Input size
    - 640x480
- Plot
    - <img src="./images/yolo-comparison-plots.png" width="700" />

For testing we will focus not only on the inference time, but also the preprocessing and postprocessing. We have to consider these steps as they are all part of the pipeline. Both models use the same input size and image. For leveraging the profiling task, ultralytics delivers metadata from the pipeline, so we will extract these processes. Alternatively, you can use [line_profiler](https://github.com/pyutils/line_profiler), this is what I typically use but it requieres extracting just the necessary pipeline from pre-defined workflows.

- Preprocess
    - <img src="images/preprocess.png" width="600" />
- Inference
    - <img src="images/inference.png" width="600" />
- Postprocess
    - <img src="images/postprocess.png" width="600" />
- Conclusion
    - Yolov8 total mean time: 6.19 ms
    - Yolov5 total mean time: 8.02 ms
    - Yolov8 might be a bit larger, but its preprocessing is significantly faster than yolov5, with a similar inference time and also slightly faster postprocessing. 
    - **Winner: YOLOV8**