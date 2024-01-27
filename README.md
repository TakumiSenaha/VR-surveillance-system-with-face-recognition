# VR-surveillance-system-with-face-recognition

<img src="/images/VR_concept_for_github.png" alt="VR_concept" width="340px">

## Requirement
* Unity 2022
* [VLC for Unity](https://www.videolan.org/developers/unity.html)
  * If you cannot build, you need to purchase paid assets
* SQLite

## Getting Started
＊The following **Uity Project Setup** must be completed.



## Uity Project Setup
1. Create a 3D project
2. Importing [VLC for Unity](https://www.videolan.org/developers/unity.html) (unitypackage)
3. Importing [face-recognition-for-VR.unitypackage](/unity/face-recognition-for-VR.unitypackage)
4. Set the video URL in minimalPlayback.cs

### Description of each Scene


## Face Recognition
* About saving face images for training.


   The [train_dir](/Python/train_dir/) directory structure is shown below, where several face images are stored. (Recognition of as few as 10 photos is possible.)
   ```
   <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
   ```
   (The name of each directory will be the name returned as a result of the judgment.)
* Then run [face_train_face_net.py](/src/face_train_face_net.py) to train faces. A file named `svm_model.pkl` is generated.
   ```
   python3.11 faceTrainFaceNet.py
   ```
* [face_classification_face_net.py](/src/face_classification_face_net.py)
   Face recognition is performed on the input video, and the face coordinates are written to SQLite.
* [faciall_image_storage.py](/src/faciall_image_storage.py)
  Code for saving face images for study from video.
  Set the label of the person you want to save for the study, and create only a directory in "train_dir".
  ```python
  name = 'test_person'
  ```
* Preconfiguration must be in a CSV file.
  [face_data.csv](/src/csv/face_data.csv)
ID,name,flag,x,y,z,profile
1,Lab_Member_1,0,0.0,0.0,0.0,He is Lab member.
2,Lab_Member_2,0,0.0,0.0,0.0,He is Lab member.