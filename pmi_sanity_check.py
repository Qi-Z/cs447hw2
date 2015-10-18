########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Validation script for PMI code (Part 2)
##
import sys, traceback

if __name__ == "__main__":
    # PMI test script
    try:
        print "Sanity-checking PMI implementation..."
        # Import your solution
        import hw2_pmi as pm
        # Read corpus
        corpus = pm.readFileToCorpus('movies.txt')
        pmi = pm.PMI(corpus)
        # Check PMI of 'vader' and 'luke'
        lvPMI = pmi.getPMI("luke", "vader")
        if lvPMI > 8.0 and lvPMI < 9.0:
            print "INFO: pmi(luke, vader) is probably correct."
        else:
            print "ERROR: pmi(luke, vader) is incorrect."
        # Check list of highest-PMI pairs:
        numPairs = 10
        k = 200
        commonWords = pmi.getVocabulary(k)    # words must appear in least k sentences
        pairs = pmi.getPairsWithMaximumPMI(commonWords, numPairs)
        (p, w1, w2) = pairs[0]
        if w1 == 'effects' and w2 == 'special':
            print "INFO: You found the correct highest-PMI pair (out of frequently-occuring words)."
        else:
            print "ERROR: You did not find the highest-PMI pair of frequent words (should be \"special effects\")."
    except Exception as e:
        # If your PMI code crashes, print exception
        print "ERROR: An exception was thrown while running your PMI code:"
        traceback.print_exc()
    else:
        # If you reach this clause, we can at least run your code
        print "SUCCESS: Your PMI code compiles and runs without throwing an unhandled exception (you may still need to verify correctness)."

