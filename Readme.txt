pip install opencv-python mss numpy gymnasium stable-baselines3 ultralytics pydirectinput selenium webdriver-manager pytesseract
https://github.com/UB-Mannheim/tesseract/wiki tesseract-ocr-w64-setup-5.5.0.20241111.exe installed to path \Program Files\Tesseract-OCR

Testing.py runs the program though selenium to launch its own browser of the game. input is not independant so requires focus on the window
and unable to multitask while testing. Screen is recored through mss screen grab, yolo model is created through roboflow to annotate images,
currently 225 images and run through a yolo8 model via train.py OCR uses Tesseract to capture score data for reward calculation. PPO
through stable-baseline3 for the reinforced learning model. optional debug to see ai views. game over is viewed as score decreasing and
penalizes the reward algorithm when it occures. reward is given for increased score. A for loop iterates and captures the coordinates for
each object detected from the yolo model and the coordinates are passed to in a flattened array. it also calculates the closest object of 
each type to the player. Current problems are that the model hallucinates that some tracks upper coordinates are part of the background
near the ui elements, creating massive bounding boxes, a stop gap at the moment is blocking these objects from being passed along as it 
would require massive work re-training the yolo model with a more expansive dataset. A better reward and failsafe against false positives
in score dropping would also help to better train the model. 
