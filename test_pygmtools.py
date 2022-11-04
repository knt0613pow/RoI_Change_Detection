import torch
import pygmtools as pygm
pygm.BACKEND = 'pytorch'
_ = torch.manual_seed(1)

# Generate a batch of isomorphic graphs
batch_size = 10
X_gt = torch.zeros(batch_size, 4, 4)
breakpoint()
X_gt[:, torch.arange(0, 4, dtype=torch.int64), torch.randperm(4)] = 1
A1 = 1. * (torch.rand(batch_size, 4, 4) > 0.5)
torch.diagonal(A1, dim1=1, dim2=2)[:] = 0 # discard self-loop edges
e_feat1 = (torch.rand(batch_size, 4, 4) * A1).unsqueeze(-1) # shape: (10, 4, 4, 1)
A2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), A1), X_gt)
e_feat2 = torch.bmm(torch.bmm(X_gt.transpose(1, 2), e_feat1.squeeze(-1)), X_gt).unsqueeze(-1)
feat1 = torch.rand(batch_size, 4, 1024) - 0.5
feat2 = torch.bmm(X_gt.transpose(1, 2), feat1)
n1 = n2 = torch.tensor([4] * batch_size)