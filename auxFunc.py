import os
import numpy as np
import scipy.io
from indexes import indexes

def loadExperimentData(subfolderName: str) -> np.ndarray:

    # Define the folder path
    script_dir = os.path.dirname(os.path.realpath(__file__))    
    dataFolder = os.path.join(script_dir, subfolderName)

    # Define the list of files to be read
    fileNames = ['grf.mat', 'hipPos.mat', 'isFlight.mat', 'kneePos.mat', 'motCurr.mat', 'pressure.mat', 'safety.mat']

    # Load the data from all files and concatenate horizontally
    allData = np.empty((0, 0))
    for fileName in fileNames:
        try:
            # Load the .mat file
            data = scipy.io.loadmat(os.path.join(dataFolder, fileName))
            # print(os.path.join(dataFolder, fileName))
            # get the underlying data from the dictionary
            currentData = data['data'][-1][-1][4]
            # print(data['data'][-1][-1][-1])

            # Concatenate the currentData array horizontally with allData
            if allData.size == 0:
                allData = currentData
            else:
                # concatenate horizontally but the last column of allData and the first column of currentData
                allData = np.hstack((allData, currentData[:, :-1]))
        except Exception as e:
            print(e)
            print(f"File {fileName} not found")

    return allData

def getStanceIntervals(allData: np.ndarray) -> np.ndarray:
    # Create a list to store the step intervals
    stanceIntervals = []

    # Get the interval data for the current interval
    isflight = allData[:, indexes['isFlight']]
    
    inFlight = True
    currentStanceStart = 0
    for i in range(len(isflight)):
        if isflight[i] == 0:
            # We are not in a stance interval, and one started now
            if inFlight:
                inFlight = False
                currentStanceStart = i
        else:
            # We are in a step interval and it ended
            if not inFlight:
                inFlight = True
                currentStanceEnd = i - 1
                
                # Check if the step interval is longer than 100 data points
                if (currentStanceEnd - currentStanceStart + 1) >= 100:
                    stanceIntervals.append([currentStanceStart, currentStanceEnd])

    stanceIntervals = np.array(stanceIntervals)

    # filter intervals where the step interval size is an outlier
    stepIntervalSizes = stanceIntervals[:, 1] - stanceIntervals[:, 0]
    stepIntervalSizeMean = np.mean(stepIntervalSizes)
    stepIntervalSizeStd = np.std(stepIntervalSizes)

    # Filter the intervals where the step interval size is an outlier
    filteredStanceIntervals = np.zeros_like(stanceIntervals)
    unfilteredIntervals = 0
    for i in range(len(stepIntervalSizes)):
        if np.abs(stepIntervalSizes[i] - stepIntervalSizeMean) < 2 * stepIntervalSizeStd:
            filteredStanceIntervals[unfilteredIntervals] = stanceIntervals[i]
            unfilteredIntervals += 1
    
    filteredStanceIntervals = np.copy(filteredStanceIntervals[:unfilteredIntervals])
    print(f"Filtered {len(stanceIntervals) - unfilteredIntervals} step intervals")
    return filteredStanceIntervals



def getStepIntervals(stanceIntervals: np.ndarray) -> np.ndarray:
    # loop through all the steps skip the first interval
    # get the start index of each interval and the first index of the next interval
    # each step will start on the stance. Then the swing phase last until the next stance on next step
    # last step is not considered
    # get the smallest interval size of a swing/flight phase interval
    swingPhaseIntervalSize = np.inf
    for i in range(0, stanceIntervals.shape[0]-1):
        flighStartIndex = stanceIntervals[i, 1]
        flightEndIndex = stanceIntervals[i+1, 0]
        currentSwingPhaseIntervalSize = flightEndIndex - flighStartIndex
        if currentSwingPhaseIntervalSize < swingPhaseIntervalSize:
            swingPhaseIntervalSize = currentSwingPhaseIntervalSize
            
    # now create the full step interval (stance and swing) for each step
    # create new array using stanceIntervcals as reference
    stepIntervals = np.zeros((stanceIntervals.shape[0], 2), dtype=int)
    for i in range(stanceIntervals.shape[0]):
        start = stanceIntervals[i, 0]
        end = stanceIntervals[i, 1] + swingPhaseIntervalSize
        stepIntervals[i, :] = [start, end]


    return stepIntervals

def getPaddedKneeAngleVectors(kneeAngle: np.ndarray, stepIntervals: np.ndarray, convertToExtensionOn180: bool = False) -> np.ndarray:
    # get all step intervals and pad them to the same size
    maxIntervalSize = np.max(stepIntervals[:, 1] - stepIntervals[:, 0])

    kneeAngleVectors = np.zeros([ stepIntervals.shape[0],maxIntervalSize])
    # looping through all step intervals
    for i in range(1,stepIntervals.shape[0]):
        kneeAngleStep = kneeAngle[stepIntervals[i, 0]:stepIntervals[i, 1]]

        # Pad the data to match the largest interval size
        paddingSize = maxIntervalSize - len(kneeAngleStep)
        paddedKneeAngleStep = np.pad(kneeAngleStep, (0, paddingSize), mode='edge')

        kneeAngleVectors[i-1] = paddedKneeAngleStep
    
    if convertToExtensionOn180:
        kneeAngleVectors = 180 - kneeAngleVectors
    
    return kneeAngleVectors