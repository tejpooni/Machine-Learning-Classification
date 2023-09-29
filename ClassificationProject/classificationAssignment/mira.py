# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            cGrid = [0.001, 0.002, 0.004, 0.008]
        else:
            cGrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, cGrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, cGrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        bestAccuracyCount = -1  # best accuracy so far on validation set
        cGrid.sort(reverse=True)
        bestParams = cGrid[0]
        "*** YOUR CODE HERE ***"
        # Implementing trainAndTune
        for c in cGrid:
            # Initialize weights
            weights = {}
            for label in self.legalLabels:
                weights[label] = util.Counter()  # initializing counter for each field in dict.

            for iteration in range(self.max_iterations):
                print("Starting iteration ", iteration, "...")
                for i in range(len(trainingData)):
                    vectors = util.Counter()

                    for l in self.legalLabels:
                        vectors[l] = weights[l] * trainingData[i]

                    pred = vectors.argMax()

                    if trainingLabels[i] != pred:
                        f = trainingData[i].copy()
                        t = min(c, ((weights[pred] - weights[trainingLabels[i]]) * f + 1.0) / (2.0 * (f * f)))

                        f.divideAll(1.0 / t)
                        weights[trainingLabels[i]] += f
                        weights[pred] -= f

            self.weights = weights

            # Classify validation data
            guesses = self.classify(validationData)

            # Calculate accuracy
            accuracyCount = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)

            if accuracyCount > bestAccuracyCount:
                bestAccuracyCount = accuracyCount
                bestParams = c
                self.weights = weights

        # util.raiseNotDefined()
            print("finished training. Best cGrid param = ", bestParams)

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        "*** YOUR CODE HERE ***"
        for datum in data:
            vectors = util.Counter()

            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum

            guesses.append(vectors.argMax())

        # util.raiseNotDefined()
        return guesses


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        # featuresWeights = []

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # return featuresWeights
        return [element for element in self.weights[label].sortedKeys()[:100]]