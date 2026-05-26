Go to kaggle to download the FER-2013 dataset of images of people's facial expressions. 
After downloading put all of the files in the same folder and rename the FER-2013 folder to fer2013
fer2013 should have two sections, test and train
Be sure to enable webcam permissions


basic commands include:

python3 expression_detector.py

python3 expression_detector.py --train --epochs 10

python3 expression_detector.py --train --epochs 20

python3 expression_detector.py --train --epochs 50

Note: On Mac m4 50 epochs take ~30 minutes to run. Without GPU and running solely on CPU 50 epochs will take ~2-3 hours
