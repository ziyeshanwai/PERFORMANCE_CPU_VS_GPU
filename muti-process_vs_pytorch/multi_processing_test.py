import multiprocessing
import time
import torch


def task_1():
    dim = 7
    A = torch.rand(dim, dim, requires_grad=False)
    b = torch.rand(dim, 1, requires_grad=False)
    x = torch.autograd.Variable(torch.rand(dim, 1), requires_grad=True)
    stop_loss = 1
    step_size = stop_loss / 3.0
    print('Loss before: %s' % (torch.norm(torch.matmul(A, x) - b)))
    for i in range(1000 * 1000):
        Δ = torch.matmul(A, x) - b
        L = torch.norm(Δ, p=2)
        L.backward()
        x.data -= step_size * x.grad.data  # step
        x.grad.data.zero_()
        if i % 10000 == 0:
            print('Loss is %s at iteration %i in task1' % (L, i))
        if abs(L) < stop_loss:
            print('It took %s iterations to achieve %s loss. in task1' % (i, step_size))
            break


def task_2():
    dim = 7
    A = torch.rand(dim, dim, requires_grad=False)
    b = torch.rand(dim, 1, requires_grad=False)
    x = torch.autograd.Variable(torch.rand(dim, 1), requires_grad=True)
    stop_loss = 1
    step_size = stop_loss / 3.0
    print('Loss before: %s' % (torch.norm(torch.matmul(A, x) - b)))
    for i in range(1000 * 1000):
        Δ = torch.matmul(A, x) - b
        L = torch.norm(Δ, p=2)
        L.backward()
        x.data -= step_size * x.grad.data  # step
        x.grad.data.zero_()
        if i % 10000 == 0:
            print('Loss is %s at iteration %i in task2' % (L, i))
        if abs(L) < stop_loss:
            print('It took %s iterations to achieve %s loss. in task2' % (i, step_size))
            break


if __name__ == "__main__":
    """"""
    """问题规模影响结果"""
    start = time.time()
    p1 = multiprocessing.Process(target=task_1, args=())
    p2 = multiprocessing.Process(target=task_2, args=())
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    end = time.time()
    print("method1 takes {}s".format(end - start))
    start = time.time()
    task_1()
    task_2()
    end = time.time()
    print("method2 takes {}s".format(end - start))
