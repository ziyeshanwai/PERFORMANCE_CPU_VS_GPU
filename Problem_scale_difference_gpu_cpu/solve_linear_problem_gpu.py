import torch
import time
"""
the code is from 
http://bytepawn.com/pytorch-basics-solving-the-axb-matrix-equation-with-gradient-descent.html
solve linear equation Ax = b
"""


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dim = 20000  # problem scale parameter
    A = torch.rand(dim, dim, requires_grad=False)
    print("变量A在cuda上:{}".format(A.is_cuda))  # 判断 A 是不是在GPU上
    b = torch.rand(dim, 1,  requires_grad=False)
    print("变量b在cuda上:{}".format(b.is_cuda))
    x = torch.autograd.Variable(torch.rand(dim, 1), requires_grad=True)
    print("变量x在cuda上:{}".format(x.is_cuda))
    stop_loss = 1
    step_size = stop_loss / 3.0
    print('Loss before: %s' % (torch.norm(torch.matmul(A, x) - b)))
    start = time.time()
    for i in range(1000*1000):
        Δ = torch.matmul(A, x) - b
        L = torch.norm(Δ, p=2)
        L.backward()
        x.data -= step_size * x.grad.data # step
        x.grad.data.zero_()
        if i % 10000 == 0:
            end = time.time()
            print('Loss is %s at iteration %i' % (L, i))
            print("it takes time:{}s".format(end - start))
        if abs(L) < stop_loss:
            print('It took %s iterations to achieve %s loss.' % (i, step_size))
            break
    end = time.time()
    print('Loss after: %s' % (torch.norm(torch.matmul(A, x) - b)))
    print("it takes time:{}s".format(end - start))