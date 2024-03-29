# VR-surveillance-system-with-face-recognition

<img src="/images/VR_concept_for_github.png" alt="VR_concept" width="340px">

## Requirement
* [Unity 2022.3.3f1](https://unity.com/releases/editor/whats-new/2022.3.3#release-notes)
* [VLC for Unity](https://www.videolan.org/developers/unity.html)
  * If you cannot build, you need to purchase paid assets
* [SQLite](https://www.sqlite.org/index.html)

## Getting Started on PC
＊The following **Uity Project Setup** must be completed.
1. After the Unity setup is complete, check that the video plays in the *PC scene*.
   - For example, you can download [/src/video/360-degreees_video_for_test.mp4](/src/video/360-degreees_video_for_test.mp4) for a video and specify the URL in *MinimalPlayback.cs*, you can play it.
  
2. Preparation for Face Recognition (A detailed description is [below](#face-recognition))
   - Facial recognition results are stored in a database, which Unity refers to for tracking.
   - The database is located in **"~/Unity project path/Assets/database/face_data.db"**.
   - Set database path in [face_classification_face_net.py](/src/face_classification_face_net.py) and Include the same video URL as in *MinimalPlayback.cs*
      ```py
      db_path = '~/Unity project path/Assets/database/face_data.db'
      video_url = "[The path you downloaded]/360-degreees_video_for_test.mp4"
      ```
      *MinimalPlayback.cs* requires the URL to be entered, so even if the video is locally available, it must be entered as "file:///[The path you downloaded]/360-degreees_video_for_test.mp4".

3. Simultaneously plays the **Scene** in Unity and executes **[face_classification_face_net.py](/src/face_classification_face_net.py)**.
   - The face recognition results (where the face is located) for the video are transmitted to Unity through the database for face tracking.


**When using local video files, even if face recognition takes time to process a frame, the next frame to be processed is set to 15 frames later, which causes a significant delay compared to the video playback in Unity.
With streaming video, frames arriving during face recognition are discarded, so frames can always be processed at the same time as the playback video. (This still causes a delay, however...).**

## Uity Project Setup
1. Create a 3D project
2. Importing [VLC for Unity](https://www.videolan.org/developers/unity.html) (unitypackage)
3. Importing [face-recognition-for-VR.unitypackage](/unity/face-recognition-for-VR.unitypackage)
4. Set the video URL in MinimalPlayback.cs
   * RTSP streaming : rtsp://192.168.xx.xx:8554
   * Local mp4 file : file:///C:/~
   * If you define a public variable in MinimalPlayback.cs, you can set the URL in "Inspector".
    ```cs
    public string URL;
        .
        .
        .
    // playing remote media
    _mediaPlayer.Media = new Media(_libVLC, new Uri(URL));
        .
        .
        .
    ```

### Description of each Scene
- **PC**
  - 360-degree video playback on a PC screen. Set up as follows.(Inspector of Sphere100)
<p align="center">
<img src="/images/Scene_PC_inspector.png" alt="Scene_PC_inspector" width="400px">
</p>


- **HMD_EXP_ASSIST**
  - For viewing 360-degree videos on a head-mounted display. As in the **PC scene**, scripts are attached to the Sphere100, but the scripts to be kept active are the same as in the **PC scene**. 
  <p align="center">
  *** Note that different scripts are checked by default ***
  </p>

## Face Recognition
<details><summary>Face Net</summary>
Use Google's Face Net for facial recognition.

>S. Florian, K. Dmitry and P. James,
``Facenet: A unified embedding for face recognition and clustering,''
2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
vol. 2015, p. 815, June 2015.
</details>

* About saving face images for training.


   The [train_dir](/Python/train_dir/) directory structure is shown below, where several face images are stored. (Recognition of as few as 1 photo is possible.)
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
  ```
  ID,name,flag,x,y,z,profile
  1,Lab_Member_1,0,0.0,0.0,0.0,He is Lab member.
  2,Lab_Member_2,0,0.0,0.0,0.0,He is Lab member.
  ```