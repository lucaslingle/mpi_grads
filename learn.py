import torch as tc
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD


class LinearRegression(tc.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = tc.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin(x)


np.random.seed(0)
tc.manual_seed(0)
NUM_EXAMPLES = 32
IN_DIM = 10
OUT_DIM = 1
data_X = tc.Tensor(np.random.normal(loc=0.0, scale=1.0, size=(NUM_EXAMPLES, IN_DIM))).float()
data_Y = tc.Tensor(np.random.normal(loc=0.0, scale=1.0, size=(NUM_EXAMPLES, OUT_DIM))).float()

model = LinearRegression(IN_DIM, OUT_DIM)
optimizer = tc.optim.SGD(model.parameters(), lr=1.0)
criterion = tc.nn.MSELoss()

model.train()
OPT_ITERS = 1
for _ in range(OPT_ITERS):
    loss = criterion(input=model(data_X), target=data_Y)
    optimizer.zero_grad()
    loss.backward()
    with tc.no_grad():
        for p in model.parameters():
            g_old = p.grad
            g_new = np.zeros_like(g_old.numpy())
            comm.Allreduce(sendbuf=g_old.numpy(), recvbuf=g_new, op=MPI.SUM)
            p.grad.copy_(tc.Tensor(g_new).float())
            print(p.grad)
    optimizer.step()
