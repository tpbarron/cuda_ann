
import sys, math

n1 = int(sys.argv[1])
n2 = int(sys.argv[2])

for i in range((n1+1)*n2):
    node1 = i%(n1+1)
    node2 = i%(n2)
    index1 = n1*node2+node1
    index2 = n2*node1+node2
    print "node1="+str(node1)+", node2="+str(node2)+", [nlayer1*node2+node1]="+str(index1)+", [nlayer2*node1+node2]="+str(index2)
