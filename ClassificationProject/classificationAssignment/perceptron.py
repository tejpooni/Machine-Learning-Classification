# perceptron.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
#

# Perceptron implementation
import random

import util
PRINT = True

class PerceptronClassifier:
    """
    Perceptron classifier.
    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector of values).
        """

        self.features = trainingData[0].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING

        for iteration in range(self.max_iterations):

            # randomized order to training process, shuffle trainingData and trainingLabels before each iteration
            # Combine, shuffle and split again
            combined = list(zip(trainingData, trainingLabels))
            random.shuffle(combined)
            trainingData, trainingLabels = zip(*combined)

            print("Starting iteration ", iteration, "...")
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                # print("perceptron: train(). to be done by students")
                vectors = util.Counter()
                #calculating scores for labels:
                for l in self.legalLabels:
                    vectors[l] = self.weights[l] * trainingData[i]
                pred = vectors.argMax()
                #find the most optimum label:
                #update weight if necessary:
                if not trainingLabels[i] == pred:
                    self.weights[trainingLabels[i]] = self.weights[trainingLabels[i]] + trainingData[i]
                    self.weights[pred] = self.weights[pred] - trainingData[i]

        print("finished training")



    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.
        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
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

