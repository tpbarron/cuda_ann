## GPU Net Assumptions

* No host function that evokes kernels will call cudaDeviceSynchronize() at the end. For example, there is a backprop function on the host that executes several kernels. The last two are weight updates. Synchronization may be necessary if reading those weights immediately from a different stream. The backprop function will NOT synchronize. 

### Notes:
* Using the face dataset which has 6977 patterns and 361 inputs to 1 output, feed_forward_1_2 required 29.3051 ms for 100 half iterations and feed_forward_2_2 required .604352 ms for 100 half iterations. 
* That means I need to be able to sum in less than 29 ms or so for there to be improvement.
* I'm having trouble passing a pointer to an index of an array and still having the reduce kernel work as expected.


NVidia GTX 660 graphics card
Theoretical peak bandwidth = Specs say 144.2 gb / sec
