
import sys
import numpy as np

def generate_set(n_patterns, n_input, n_output):
    rand_set = []
    for i in range(n_patterns):
        inputs = np.random.randint(100, size=n_input) #generate random list of length n_input with integers between 0 and 100 
        outputs = np.random.randint(100, size=n_output)
        rand_set.append((inputs, outputs))
    return rand_set



DIR = "datasets/"

def write_set(rand_set, n_patterns, n_input, n_output):
    f = open(DIR+str(n_patterns)+"_"+str(n_input)+"_"+str(n_output)+".dat", 'w')
    #f.write(str(n_patterns)+" "+str(n_input)+" "+str(n_output)+"\n")
    f.write(str(n_input+n_output)+"\n")
    for p in rand_set:
        for i in p[0]:
            f.write(str(i) + " ")
        #f.write("\n")
        for o in p[1]:
            f.write(str(o) + " ")
        f.write("\n")
    f.flush()
    f.close()            
            
            
if __name__ == "__main__":
    n_patterns = int(sys.argv[1])
    n_input = int(sys.argv[2])
    n_output = int(sys.argv[3])
    
    rand_set = generate_set(n_patterns, n_input, n_output)
    
    write_set(rand_set, n_patterns, n_input, n_output)
