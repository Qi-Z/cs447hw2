########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Train a bigram HMM for POS tagging
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from math import log

# Unknown word token
UNK = 'UNK'


# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]


# Class definition for a bigram HMM
class HMM:
    ### Helper file I/O methods ###
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
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

    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r")  # open the input file in read-only mode
            sens = [];
            for line in file:
                sentence = line.split()  # split the line into a list of words
                sens.append(sentence)  # append this list as an element to the list of sentences
            return sens
        else:
            print "Error: unlabeled data file", inputFile, "does not exist"  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit()  # exit the script
            ### End file I/O methods ###

    # Constructor
    def __init__(self, unknownWordThreshold=5):
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        # Initialize the rest of your data structures here ###
        self.pi = defaultdict(float)
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.emissions = defaultdict(lambda: defaultdict(float))
        self.trellis = defaultdict(lambda: defaultdict(float))
        self.lexicon = defaultdict(lambda: set([]))  # A list of words together with their possible tags.
        self.tag_counts = defaultdict(float)
        self.vocab = set()
        self.start_tag_count = 0.0

    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        data = self.readLabeledData(trainFile)  # data is a nested list of TaggedWords
        print "Your first task is to train a bigram HMM tagger from an input file of POS-tagged text"
        data = self.preprocess(data, self.minFreq)
        self.pi_tag_transition_emission(data)
        self.start_tag_counting()
        self.replace_with_prob_matrix()

    # Calculate pi vector, C(t_i), P(t_i|t_i-1) and P(w_i | t_i)
    def pi_tag_transition_emission(self, data):
        for sen in data:
            # Initialize initial state vector.
            self.pi[sen[0].tag] += 1.0  # PI_t_i
            self.tag_counts[sen[0].tag] += 1.0  # C(t_i)
            self.emissions[sen[0].tag][sen[0].word] += 1.0  # P(w_i|t_i)
            self.lexicon[sen[0].word].add(sen[0].tag)

            for idx in range(1, len(sen)):
                self.transitions[sen[idx - 1].tag][sen[idx].tag] += 1.0  # P(t_i|t_i-1)
                self.tag_counts[sen[idx].tag] += 1.0  # C(t_i)
                self.emissions[sen[idx].tag][sen[idx].word] += 1.0  # P(w_i|t_i)
                self.lexicon[sen[idx].word].add(sen[idx].tag)

    def transition_prob(self, prev_tag, current_tag):  # with add-one smoothing, and natural log prob.
        prev_count = self.tag_counts[prev_tag]
        tag_vocab = float(len(self.tag_counts))
        return log((1.0 + self.transitions[prev_tag][current_tag]) / (prev_count + tag_vocab))

    def emission_prob(self, tag, word):
        word_count = 0.0
        for w, c in self.emissions[tag].iteritems():
            word_count += c
        if self.emissions[tag][word] == 0.0:
            return float('-inf')
        return log(self.emissions[tag][word])  # 1st element 1 means non-zero prob.

    def start_tag_counting(self):
        self.start_tag_count = 0.0
        for tag, c in self.pi.iteritems():
            self.start_tag_count += c

    def pi_prob(self, tag):
        if self.pi[tag] == 0.0:
            return float('-inf')  # Return a tuple so that if it is 0, we can save some computation.
        return log(self.pi[tag] / self.start_tag_count)

    def pretty_print(self, data):
        for sen in data:
            for w in sen:
                print(w.word, w.tag)
            print("\n")

    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):
        data = self.readUnlabeledData(testFile)
        file = open(outFile, 'w+')
        # Preprocess test
        data = self.preprocess_test(data)
        for sen in data:
            # p = self.viterbi(sen)
            # print p

            vitTags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i] + "_" + vitTags[i] + " "
            print senString
            print >> file, senString.rstrip()

    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags 
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        print "Your second task is to implement the Viterbi algorithm for the HMM tagger"
        # returns the list of Viterbi POS tags (strings)
        # Prepocess
        words = self.preprocess_test_sentence(words)

        trellis = defaultdict(lambda: defaultdict(lambda: float('-inf')))
        backpointer = defaultdict(lambda: defaultdict())
        for t in self.pi.keys():
            if self.pi[t] != float('-inf'):
                if self.emissions[t][words[0]] != float('-inf'):
                    trellis[t][0] = self.pi[t] + self.emissions[t][words[0]]
                    backpointer[t][0] = 'BEGINTAG'
        for i in range(1, len(words)):
            for t in self.lexicon[words[i]]:
                # Only check the possible tags for w_(i-1)
                candicates_trellis = []
                candicates_ptr = []
                for prev_t in self.lexicon[words[i - 1]]:
                    if self.emissions[t][words[i]] != float('-inf'):
                        candicates_trellis.append((trellis[prev_t][i - 1] + self.transitions[prev_t][t] +
                                                   self.emissions[t][words[i]], prev_t))
                        candicates_ptr.append((trellis[prev_t][i - 1] + self.transitions[prev_t][t], prev_t))

                trellis[t][i] = max(candicates_trellis)[0]
                backpointer[t][i] = max(candicates_ptr)[1]



        candidates = []
        for t in self.lexicon[words[-1]]:
            candidates.append((trellis[t][len(words) - 1], t))
        last_tag = max(candidates)[1]

        tags = []
        prev_tag = last_tag
        for i in reversed(range(1, len(words))):
            prev_tag = backpointer[prev_tag][i]
            tags.insert(0, prev_tag)

        tags.append(last_tag)
        return tags

        # return ["NULL"] * len(words)  # this returns a dummy list of "NULL", equal in length to words

    # Replace with UNK, get Vocabulary.
    def preprocess(self, data, threshold):
        freqDict = defaultdict(int)
        for sen in data:
            for w in sen:
                freqDict[w.word] += 1

        for sen in data:
            for i in range(0, len(sen)):
                w = sen[i]
                if freqDict[w.word] < threshold:
                    sen[i].word = UNK
        self.vocab = set([tw.word for sentence in data for tw in sentence])
        return data

    # Replace test data with UNK.
    def preprocess_test(self, data):
        for sen in data:
            for i in range(0, len(sen)):
                word = sen[i]
                if word not in self.vocab:
                    sen[i] = UNK
        return data

    def preprocess_test_sentence(self, sen):
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in self.vocab:
                sen[i] = UNK
        return sen

    def replace_with_prob_matrix(self):
        tmp = defaultdict(lambda: float('-inf'))
        for t, c in self.pi.iteritems():
            prob = c/self.start_tag_count
            if prob == 0.0:
                tmp[t] = float('-inf')
            else:
                tmp[t] = log(prob)
        self.pi = tmp
        tmp = defaultdict(lambda: defaultdict(lambda: float('-inf')))
        for prev_t, d in self.transitions.iteritems():
            for cur_t, c in d.iteritems():
                prev_count = self.tag_counts[prev_t]
                tag_vocab = float(len(self.tag_counts))
                tmp[prev_t][cur_t] = log((1.0 + self.transitions[prev_t][cur_t]) / (prev_count + tag_vocab))
        self.transitions = tmp
        tmp = defaultdict(lambda: defaultdict(lambda: float('-inf')))
        for t, d in self.emissions.iteritems():
            for w, c in d.iteritems():
                tmp[t][w] = self.emission_prob(t, w)
        self.emissions = tmp


if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')
