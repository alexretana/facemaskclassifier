import pandas as pd
from sklearn.model_selection import train_test_split
import os
from imutils import paths
import shutil
from tqdm import tqdm

#######################################
# Variable Constants #
ORIG_INPUT_DATASET = "mask-classifer-images"
BASE_PATH = "dataset"

TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

CLASSES = ["Mask", "No_Mask"]

BATCH_SIZE = 32

#########################################

#opens the file with list of training image paths.
#creates dataframe with path and label index
def openTrainList():
    with open('imglist_train.txt', 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            lines[idx] = line.split()
            
        df = pd.DataFrame(lines, columns=["imgpath", "label"])

    #assign series to X and Y
    X = df["imgpath"]
    Y = df["label"]

    #create training and validation split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_val, Y_train, Y_val

#open the file for test image paths. creates similar df as for training data
def openTestList():
    with open('imglist_test.txt') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            lines[idx] = line.split()
        
        df = pd.DataFrame(lines, columns=["imgpath", "label"])

    #assign series to X_test and Y_test    
    X_test, Y_test = df["imgpath"], df["label"]

    return X_test, Y_test

#creates dictionary of each split, nested with data
def makePathDict(X_train, Y_train, X_test, Y_test, X_val, Y_val):
    pathDict = {TRAIN:{'X':X_train,'Y':Y_train},
                TEST:{'X':X_test,'Y':Y_test},
                VAL:{'X':X_val,'Y':Y_val}}

    return pathDict

#function that actually does the copying
def makeDatasetDir(pathDict):
    #these loops copy the images to a folder with a new architecture
    for split, val in pathDict.items():
        print("[INFO] processing '{} split'...".format(split))
        
        #loop through each filename, create new file path, copy image to new path
        for ind, x in tqdm(val['X'].items()):
            filename = "_".join(x.split("/")[-2:])
            label = CLASSES[int(val['Y'][ind])]
            
            dirPath = os.path.sep.join([BASE_PATH, split, label])
            
            #make dir if it doesn't exist
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            
            #copy
            p = os.path.sep.join([dirPath, filename])
            shutil.copy2(x, p)

if __name__ == '__main__':

    X_train, X_val, Y_train, Y_val = openTrainList()
    X_test, Y_test = openTestList()
    pathDict = makePathDict(X_train, Y_train, X_test, Y_test, X_val, Y_val)
    makeDatasetDir(pathDict)