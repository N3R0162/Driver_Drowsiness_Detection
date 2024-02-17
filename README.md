# Driver Drowsiness Detection using Eye Landmarks

## Description
This application detects driver drowsiness based on the condition of their eyes. It utilizes eye landmarks to assess the level of drowsiness and provide alerts when necessary.

# Auto Installation
The set up process can be done by running the following command after create a virtual environment:
```bash 
sh setup.sh
```
Or the manual installation can be done by following the steps below.

## Manual Installatio
To install the required dependencies, use the following command:
```bash
pip3 install -r requirements.txt
```

### This project use FaceBoxV2 for detecting facial bounding box, in order for the app to run, first the FaceBoxV2 must be built:
```bash
cd ./FaceBoxesV2/utils/
sh make.sh
```

## To run the program, execute the following command:
```bash
streamlit run ./source/main.py
```

## License:
This project is licensed under the terms of the MIT License. See the LICENSE file for details.
