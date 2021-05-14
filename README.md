# mpi_grads

Use mpi4py to perform multi-process gradient computation. Proof of concept!
Below is an example. In this simple example, all data is available on all processes, and the gradients for the second script are summed rather than averaged.
As we can see, the resulting gradients printed by both processes are the sum of those we'd get for only a single process! 

```
(openai_gym) lucaslingle@Lucass-MacBook-Pro mpi_grads % python learn.py
tensor([[-0.7045,  0.6553, -1.0252, -0.2712, -0.1706,  0.6227,  0.0923,  0.0981,
         -0.4659, -0.4867]])
tensor([0.4469])
(openai_gym) lucaslingle@Lucass-MacBook-Pro mpi_grads % mpirun -np 2 python -m learn_mpi
tensor([[-1.4090,  1.3106, -2.0504, -0.5424, -0.3413,  1.2454,  0.1846,  0.1962,
         -0.9317, -0.9734]])
tensor([[-1.4090,  1.3106, -2.0504, -0.5424, -0.3413,  1.2454,  0.1846,  0.1962,
         -0.9317, -0.9734]])
tensor([0.8939])
tensor([0.8939])
```

To average is easy, we can just divide the result through by ```MPI.COMM_WORLD.Get_size()``` before assigning the gradients.
