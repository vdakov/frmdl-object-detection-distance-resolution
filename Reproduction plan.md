*Group 15* - Vasil Dakov, Lauri Warsen, Laszlo Roovers

## Tasks + Division 
*An overview of what has to be done to reproduce + extend the paper.*

(Minimum) **Goal:** Reproduce Fig. 7 ![[Pasted image 20250521102714.png]]

#### Paper Model Implementation 
***Input***: YOLOv5 implementation by Ultralytics https://github.com/ultralytics/yolov5/blob/master/models/yolo.pyen
***Output**:* RA-YOLO implementation (+ YOLO-3, YOLO-4, YOLO-5, not to be confused with YOLOv3, YOLOv4, etc. ) 
***Description**:* Implement the RA-YOLOv5 by taking the different modular components from the original Ultralytics codebase. Gives an implementation of the model used in the paper including all of its special components.
***Assignee**:* Vasil Dakov 
***Status***: Work-In-Progress

#### Model Fine-Tuning and Training 
***Input*** : RA-YOLO implementation + YOLO-3, YOLO-4, YOLO-5
***Output**:* Pre-trained + Re-trained/Fine-tuned models on the Processed data
***Description**:* Retrieve the Provides us with models we can run on the test datasets so we can reproduce the Recall vs. Distance figures from the original paper
***Assignee**:* Lauri
***Status***: Work-In-Progress

#### Original Dataset Acquisition
***Input***: URLs/names of datasets (TJU, EuroCity)
	- TJU-DHD: https://github.com/tjubiit/TJU-DHD 
	- EuroCity: https://eurocity-dataset.tudelft.nl/
***Output***: Local copies of datasets with annotations
***Description***: Download and organize the datasets. Ensure annotations are formatted uniformly for YOLO training (e.g., pedestrian and rider bounding boxes).
***Assignee:*** Vasil Dakov
***Status***: Complete

#### Data Processing - Spatial Downsampling 
***Input***: Original high-resolution images (from TJU, EuroCity)
***Output***: Down-sampled images at target scales (Down2, Down4 for TJU; Down1.42 for EuroCity)
***Description***: Use FFmpeg (with `bicubic` filter) to produce images at reduced resolutions as done in the paper, to study spatial resolution effects.
***Assignee:*** Lauri
***Status***: Work-In-Progress
#### Data Processing - Amplitude Compression
***Input***: Original and down sampled images
***Output***: BPG-compressed images with QP values ranging from 0 to 51
***Description***: Apply the BPG encoder to compress images with varying quantization (amplitude) resolution. These will be used for robustness experiments.
***Assignee:*** Laszlo
***Status***: Work-In-Progress

#### Data Processing - Eurocity Distance Binning
***Input***: Eurocity Dataset
***Output***: Dataset binned in groups of 10m
***Description***: Analyze the existing annotations of Eurocity. Split the labels into groups and extend the label format.
***Assignee:*** Laszlo
***Status***: Work-In-Progress
#### Mixed-Resolution Dataset Preparation
***Input***: All processed images (original, downsampled, compressed) with annotations
***Output***: Dataset ready for training with varying spatial and amplitude resolutions
***Description***: Merge processed data into a cohesive dataset with consistent annotations and labels for training RA-YOLO and baseline models.
***Assignee:*** Laszlo
***Status***: Work-In-Progress

#### Control Dataset Acquisition 
***Input***: None, we create it. (Minecraft, I guess?)
***Output:*** A control dataset of images in Minecraft, at different block distances, and everything else identical.
***Description:*** Automate a procedure to take screenshots of the same object/s in minecraft at different (identical) block distance. Annotate them manually, in the same format as the other datasets. 
***Assignee:*** Vasil
***Status***: Work-In-Progress

#### Data Visualization Code
***Input**:* Toy data to test plots.
***Output**:* Plots in the same format as the original paper.
***Description**:* Python code to get out the same plots. Test the code with sample toy data.
***Assignee:*** Laszlo
***Status***: Work-In-Progress

#### Evaluation + Visualization
 ***Input**: Trained models, test sets (varied by resolution and QP), EuroCity distances
***Output***:
    - Graphs: mAP vs PSNR, mAP vs image size, Recall vs distance
    - Visual detection comparisons, similar to the ones in the paper
***Description***: Recreate plots from Figure  7 with the new data . Evaluate how performance varies with resolution, compression, and object distance. Include side-by-side sample detections.
***Assignee:*** Lauri
***Status***: Work-In-Progress

### Report and Discussion
**Input**: All experimental results, visualizations, implementation notes  
**Output**: Final structured reproduction report  
**Description**: Write and organize the final report with detailed commentary, challenges, results vs. original, and insights.  
**Assignee**:  All
**Status**: Not started



## Timeline

##### Weeks 1 - 4: Literature review and theory preparation 
- General object detection literature: 
	- R-CNN (https://arxiv.org/pdf/1311.2524)
	- Fast R-CNN (https://arxiv.org/abs/1504.08083)
	- Faster R-CNN (https://arxiv.org/pdf/1506.01497)
	- YOLOv1 (https://arxiv.org/pdf/1506.02640)
	- YOLOv2 (https://arxiv.org/abs/1612.08242), 
	- YOLOv3 (https://arxiv.org/abs/1804.02767), 
	- YOLOv5 (https://docs.ultralytics.com/models/yolov5/)
	- DarkNet: https://paperswithcode.com/method/darknet-19 
	- Feature Pyramid Networks for Object Detection - (https://arxiv.org/abs/1612.03144)
- Resolution: 
	- BPG Codec: https://en.wikipedia.org/wiki/Better_Portable_Graphics 
- Started **Paper Model Implementation** 
- Started Finished **Original Dataset Acquisition**
##### Week 5: Implementation 
- Finish **Paper Model Implementation** 
- Start and Finish **Data Processing - Spatial Downsampling
- Start and Finish **Data Processing - Amplitude Compression**
##### Week 6: Control Data, Experiment Setup 
- Start and Finish **Data Processing - Eurocity Distance Binning**
- Finish **Mixed Dataset Creation**
- Finish **Control Dataset Acquisition** 
- Start **Model Fine-Tuning and Training**
##### Week 7 - Finish Training, Start Evaluation 
- Finish **Model Fine-Tuning and Training**
- Finish **Data Visualization Code**
- Start **Evaluation + Visualization**
##### Week 8 : Report, Finish Evaluation 
- Finish **Evaluation + Visualization**
- Start and Finish **Report and Discussion**




