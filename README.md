# DanceAPP — How to Start?
(版权所有：NUS—summerworkshop-group2)
## Inspiration from the `danceapp.py` Demo

- **Test Issues**:
  - Keypoint loss in fast moves (~30% frames): incomplete keypoints (e.g., arm blocking)
  - Multi-person scenes: can’t distinguish the main dancer

- **Demo**:
  - Simple GUI (reference video + webcam)
  - Pose keypoint processing using YOLOv8n
  - → Inspired “detection → interactive feedback” framework (like *Just Dance*)

## Should We Continue to Use YOLO?

Maybe we need to test it with different models ourselves.

## Model & Parameter Setup
<img width="264" height="385" alt="image" src="https://github.com/user-attachments/assets/eba07931-c1f2-4d15-b14e-f38b044259ae" />
<img width="606" height="173" alt="image" src="https://github.com/user-attachments/assets/a32059c1-b362-41ea-ad9c-1be7e2814b39" />

After trying different parameters, we reverted to:

```python
num_poses = 1
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5
```

## Model & Parameter Setup
<img width="173" height="250" alt="image" src="https://github.com/user-attachments/assets/ed1ebba6-ec3f-4f5e-b7d1-f850453f4092" />
<img width="170" height="251" alt="image" src="https://github.com/user-attachments/assets/38688fb0-fed7-473c-9072-6ac754430d94" />
<img width="177" height="249" alt="image" src="https://github.com/user-attachments/assets/45d203d1-c95b-47c1-8498-32ba7c959ab0" />
<img width="170" height="249" alt="image" src="https://github.com/user-attachments/assets/5783fe2a-63cc-41ec-9750-2ec7c4c04d4a" />

Comparison of performance metrics for different pose detection models:

| Model   | Accuracy (AP, AR) | Speed (it/s) | Notes                          |
|---------|-------------------|--------------|--------------------------------|
| Origin  | —                 | —            |                                |
| Alpha   | —                 | —            |                                |
| MP      | —                 | —            |                                |
| YOLO    | —                 | —            |                                |

> Helps select a model for the DanceAPP.
<img width="170" height="249" alt="image" src="https://github.com/user-attachments/assets/5389111c-9da4-4ae4-a2e8-051c621482c5" />

## Detector Integration: Video Stuttering & Jitter
<img width="220" height="188" alt="image" src="https://github.com/user-attachments/assets/ee45d364-aaf5-4d3d-b340-1582f6fd6815" />
<img width="241" height="196" alt="image" src="https://github.com/user-attachments/assets/d79f245c-44b0-443d-91db-a16a59df976f" />
<img width="260" height="140" alt="image" src="https://github.com/user-attachments/assets/bee35f5a-003c-4098-be55-7f7c549a17aa" />

We tried:

- EWMA smoother (tried)
- Kalman filter (**chosen**)

**Both worked well!**

## Mediapipe Challenges

**Imported and parameters set**, then faced:

- YOLO
- MP (not improved)

> The actual quality is clear; it's just compressed in the GIF causing pixel loss.

**Motion Model**: prediction  
**New** `DetectionResults`: updated

## Detector Integration: Multi-Person Failure

Mediapipe performed poorly in multi-person scenarios.
<img width="227" height="313" alt="image" src="https://github.com/user-attachments/assets/9ecd40aa-6112-42cd-a3fd-23a8f239630e" />
<img width="233" height="312" alt="image" src="https://github.com/user-attachments/assets/0b71375b-7bc0-408d-8e58-8789a57b9a24" />

**Solution**:
- Record keypoints of the main dancer
- Optimize tracking

## Detector Integration: Occlusion
<img width="370" height="243" alt="image" src="https://github.com/user-attachments/assets/59c4bbdc-7fc8-4ac2-8ccd-cd6fb06e6a2b" />

**Core Challenge**:  
Occlusion (e.g., arms blocking torso, background clutter) caused incomplete keypoints → erratic pose tracking and unreliable scoring.

**Tried**:  
- Linear Interpolation  
- Symmetrical Mapping  

**Result**: Skipping

## Hand Keypoint Support for Gesture Dances
<img width="324" height="113" alt="image" src="https://github.com/user-attachments/assets/14a2c96c-32db-449e-bfd2-fd9ecbf6c6d2" />
<img width="108" height="100" alt="image" src="https://github.com/user-attachments/assets/dcf2f49a-98df-4ee5-b12f-0ac489e16cad" />

### Before

<img width="232" height="324" alt="image" src="https://github.com/user-attachments/assets/c9412d55-0708-4f24-b28c-a98d31a2a04b" />


### After

<img width="256" height="324" alt="image" src="https://github.com/user-attachments/assets/bd8e25bf-cf07-4d05-a7ab-0f3d08d046ba" />


## Detector Completed
<img width="598" height="347" alt="image" src="https://github.com/user-attachments/assets/db609768-dd43-4bdb-b0c5-5fe4dc4776ee" />
<img width="305" height="346" alt="image" src="https://github.com/user-attachments/assets/d05e9832-bfcc-42be-92df-6fdfaf44694f" />

**Our DanceApp — DanceVibe**

# Overview of Our Scoring System

## Difficulty Levels

- Easy  
- Medium  
- Hard  
- Expert  

## POSE / GESTURE - Precision

- Penalizes errors in:
  - Joint positions
  - Angles
  - Limb alignment

## BEAT - Musicality

- Rewards timing synced to beats

## OBSTACLE - Reactivity

- Acknowledges user response to dynamic challenges

# Scoring System
<img width="1632" height="914" alt="image" src="https://github.com/user-attachments/assets/025e8c16-b788-4431-8b5d-cc540b41141e" />
<img width="1636" height="898" alt="image" src="https://github.com/user-attachments/assets/f7d6a904-f7b0-4314-bab2-ab63a02cc208" />

### How to Calculate Similarity?

1. **Angle to Angle** – Partial angles  
2. **Point to Point** – Euclidean distance  
   - Based on `max_reasonable_distance`  
3. **Line to Line** – Cosine vector

# Scoring System: Beat

# Scoring System: Gesture

**DTW** (Dynamic Time Warping)

- Scores gesture similarity by aligning user hand movements with reference  
- Captures nuances like:
  - ‘Finger snaps’
  - ‘Wave shapes’

> Uses a **distance matrix** and **cumulative cost** to align user performance across varying tempos.

# Scoring System: Extra
<img width="361" height="339" alt="image" src="https://github.com/user-attachments/assets/abe3337a-23d3-48b0-b495-044f85230747" />
<img width="363" height="346" alt="image" src="https://github.com/user-attachments/assets/9b42811e-516d-4050-9eaa-a63cd4579a10" />

- **Reward**: Face landmarks fully inside the box  
- **Penalty**: Block if hands/feet enter the box

## Overall HTML ScreenShot
<img width="624" height="356" alt="image" src="https://github.com/user-attachments/assets/cd5b4a7b-54b5-46be-8dcf-4a4165facf3d" />







