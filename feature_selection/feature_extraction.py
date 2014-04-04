# extract features using SVD algorithm

from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.preprocessing import Normalizer

import sys
import numpy
from scipy.linalg import svd

outputs = []
inputs = []

def load_data(dfile):
    ''' Load dataset into numpy matrix'''
    f = open(dfile, 'r')
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        line_arr = [float(x) for x in line.split()]
        if (i % 2 == 1):
            inputs.append(line_arr)
            # print line_arr
        elif (i != 0):
            outputs.append(line_arr)

    f.close()
    m = numpy.array(inputs)
    return m


def write_data(dfile, m):
    f = open(dfile+".extract", 'w')
    n_patterns = len(outputs)
    n_input = m.shape[1]
    n_output = len(outputs[0])

    f.write(str(n_patterns) + " " + str(n_input) + " " + str(n_output) + "\n")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            f.write(str(m[i,j]) + " ")
        f.write("\n")
        for k in range(n_output):
            f.write(str(outputs[i][k]) + " ")
        f.write("\n")

    f.close()


def k_rank_reduction(U, k):
    # kth largest singular value is always in the kth-1 index of s
    #s_k = numpy.zeros(k)

    # k singular vectors each with same num rows as orig
    U_k = numpy.zeros((U.shape[0], k))
    # print "U_k1:", U_k

    # k singular vectors each with same num cols as orig
    #V_k = numpy.zeros((k, V.shape[1]))

    for i in range(k):
        #self.s_k[i] = self.s[i]
        # set the column of U_k
        U_k[:,i] = U[:,i]
        # set the row of V_k
        #self.V_k[i] = self.V[i]

    return U_k

def process_svd(m):
    U, s, V = svd(m, full_matrices=False)
    return U, s, V


def RFE(m):
    X = m
    y = outputs

    print "X:", X
    print "y:", y
    # Create the RFE object and rank each pixel                                                          
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    print rfe.ranking_


def norm(m):
    X_normalized = Normalizer(m)
    print X_normalized

if __name__ == "__main__":
    dfile = sys.argv[1]
    k = int(sys.argv[2])

    m = load_data(dfile)
    norm(m)
    sys.exit(1)

    #RFE(m)
    #print m
    print "m:",m.size, m.shape
    U, s, V = process_svd(m)
    #print V
    U_k = k_rank_reduction(U, k)
    print "U_k:", U_k.shape
    #print "U:", U.shape
    #print "s:", s
    #print "V:", V.shape
    write_data(dfile, U_k)
