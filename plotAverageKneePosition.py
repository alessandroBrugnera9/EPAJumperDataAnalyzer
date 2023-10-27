from matplotlib import pyplot as plt
import auxFunc
import numpy as np
from indexes import indexes

# send subfolderName = '10131426' to the function and get raw experiment data
subfolderName = '10131426'
expData = auxFunc.loadExperimentData(subfolderName)

# getting main data from experiment
timeVector = expData[:,indexes['time']]
kneeAngle = expData[:,indexes['kneeAngle']]

# treating data to plot intervals of interest
stanceIntervals = auxFunc.getStanceIntervals(expData)
stepintervals = auxFunc.getStepIntervals(stanceIntervals)
kneeAngleVectors = auxFunc.getPaddedKneeAngleVectors(kneeAngle, stepintervals, convertToExtensionOn180=True)
# create time interval for plotting by counting the number of elements at each kneeAngleVector row, and using the sampling frequency of 1000Hz
timeInterval = np.linspace(0, kneeAngleVectors.shape[1]/1000, kneeAngleVectors.shape[1])


# treat each knee angle vector as individual step, average them and plot with confidence level
meanKneeAngles = np.mean(kneeAngleVectors, axis=0)
stdKneeAngles = np.std(kneeAngleVectors, axis=0)
n_experiments = kneeAngleVectors.shape[0]

# Calculate confidence intervals (assuming a 95% confidence level)
confidence_intervals = 1.96 * (stdKneeAngles / np.sqrt(n_experiments))

# Plotting
plt.figure(figsize=(10, 6))

# Plot the mean knee angles
plt.plot(timeInterval, meanKneeAngles, label='Mean Angle', color='blue')

# Plot the confidence intervals
plt.fill_between(timeInterval, meanKneeAngles - confidence_intervals, meanKneeAngles + confidence_intervals, color='gray', alpha=0.5, label='Confidence Interval (95%)')

plt.xlabel('Time Interval [s]')
plt.ylabel('Knee Angle [deg]')
plt.title('Average Knee angle during hop cycle')
plt.legend()
plt.show()