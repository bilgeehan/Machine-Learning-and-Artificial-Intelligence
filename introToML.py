import numpy as np
import matplotlib.pyplot as plt

docData = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 0])
testData = np.array([0.33, 0.18, 0.12, 0.95, 0.18, 0.75, 0.99, 0.05, 0.22, 0.15])

iteration = 0.1
sensitivity = np.zeros(len(np.arange(0, 1, iteration)))
specificity = np.zeros(len(np.arange(0, 1, iteration)))
for i in np.arange(0, 1, iteration):
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0

    threshold = testData >= i
    for k in range(0, len(docData)):
        if docData[k] == 1 and threshold[k] == True:
            truePositive += 1
        if docData[k] == 0 and threshold[k] == False:
            trueNegative += 1
        if docData[k] == 0 and threshold[k] == True:
            falsePositive += 1
        if docData[k] == 1 and threshold[k] == False:
            falseNegative += 1

print("Test Data: " + str(testData))
print("True Positive: "+str(truePositive))
print("True Negative: "+str(trueNegative))
print("False Positive: "+str(falsePositive))
print("False Negative: "+str(falseNegative))

sensitivity = truePositive / (truePositive + falseNegative)
specificity = trueNegative / (trueNegative + falsePositive)
truePositiveRate = sensitivity
falsePositiveRate = 1 - specificity
plt.plot(falsePositiveRate, truePositiveRate)



