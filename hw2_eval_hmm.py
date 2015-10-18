########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Evaluate the output of your bigram HMM POS tagger
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_')
        self.word = parts[0]
        self.tag = parts[1]


# A class for evaluating POS-tagged data
class Eval:
    def __init__(self, goldFile, testFile):
        print "Your task is to implement an evaluation program for POS tagging"
        self.correct = self.readLabeledData(goldFile)
        self.predict = self.readLabeledData(testFile)

    def getTokenAccuracy(self):
        print "Return the percentage of correctly-labeled tokens"
        right_count = 0.0
        wrong_count = 0.0
        for sen_idx in range(0, len(self.correct)):
            for word_idx in range(0, len(self.correct[sen_idx])):
                if self.correct[sen_idx][word_idx].tag == self.predict[sen_idx][word_idx].tag:
                    right_count += 1.0
                else:
                    wrong_count += 1.0

        return right_count/(wrong_count+right_count)
    
    def getSentenceAccuracy(self):
        print "Return the percentage of sentences where every word is correctly labeled"
        right_count = 0.0
        wrong_count = 0.0
        for sen_idx in range(0, len(self.correct)):
            flag = True
            for word_idx in range(0, len(self.correct[sen_idx])):
                if self.correct[sen_idx][word_idx].tag != self.predict[sen_idx][word_idx].tag:
                    flag = False
            if flag:
                right_count += 1.0
            else:
                wrong_count += 1.0

        return right_count/(wrong_count+right_count)

    
    # Write a confusion matrix to file
    def writeConfusionMatrix(self, outFile):
        print "Write a confusion matrix to outFile; elements in the matrix can be frequencies (you don't need to normalize)"
        matrix_file = open(outFile, 'w+')
        conf_matrix = defaultdict(lambda: defaultdict(int))
        for sen_idx in range(0, len(self.correct)):
            for word_idx in range(0, len(self.correct[sen_idx])):
                conf_matrix[self.correct[sen_idx][word_idx].tag][self.predict[sen_idx][word_idx].tag] += 1

        tags = conf_matrix.keys()

        for g_t in tags:
            row_string = g_t+"\t\t"
            for p_t in tags:
                row_string += repr(conf_matrix[g_t][p_t])+"("+p_t+")\t  "
            print >> matrix_file, row_string


    # Return the tagger's precision on predicted tag t_i
    def getPrecision(self, tagTi):
        print "Return the tagger's precision when predicting tag t_i"
        TP = 0.0
        FP = 0.0
        for sen_idx in range(0, len(self.correct)):
            for word_idx in range(0, len(self.correct[sen_idx])):
                if self.predict[sen_idx][word_idx].tag == tagTi:
                    if self.correct[sen_idx][word_idx].tag == tagTi:
                        TP += 1.0
                    else:
                        FP += 1.0

        return TP/(TP+FP)

    # Return the tagger's recall on gold tag t_j
    def getRecall(self, tagTj):
        print "Return the tagger's recall for correctly predicting gold tag t_j"
        TP = 0.0
        FN = 0.0
        for sen_idx in range(0, len(self.correct)):
            for word_idx in range(0, len(self.correct[sen_idx])):
                if self.predict[sen_idx][word_idx].tag == tagTj:
                    if self.correct[sen_idx][word_idx].tag == tagTj:
                        TP += 1.0
                else:
                    if self.correct[sen_idx][word_idx].tag == tagTj:
                        FN += 1.0

        return TP/(TP+FN)

    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r")  # open the input file in read-only mode
            sens = [];
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence)  # append this list as an element to the list of sentences
            return sens
        else:
            print "Error: unlabeled data file", inputFile, "does not exist"  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit()  # exit the script

if __name__ == "__main__":
    # Pass in the gold and test POS-tagged data as arguments
    if len(sys.argv) < 2:
        print "Call hw2_eval_hmm.py with two arguments: gold.txt and test.txt"
    else:
        gold = sys.argv[1]
        test = sys.argv[2]
        # You need to implement the evaluation class
        eval = Eval(gold, test)
        # Calculate accuracy (sentence and token level)
        print "Token accuracy: ", eval.getTokenAccuracy()
        print "Sentence accuracy: ", eval.getSentenceAccuracy()
        # Calculate recall and precision
        print "Recall on tag NNP: ", eval.getRecall('NNP')
        print "Precision for tag NNP: ", eval.getPrecision('NNP')
        # Write a confusion matrix
        eval.writeConfusionMatrix("conf_matrix.txt")
