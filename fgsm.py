import torch
import torch.nn as nn
import numpy as np

# fix seed so that random initialization always performs the same 
torch.manual_seed(13)


# create the model N as described in the question
N = nn.Sequential(nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 10, bias=False),
                  nn.ReLU(),
                  nn.Linear(10, 3, bias=False))

# random input
x = torch.rand((1,10)) # the first dimension is the batch size; the following dimensions the actual dimension of the data
x.requires_grad_() # this is required so we can compute the gradient w.r.t x
t = 1 # target class

epsReal = 1.14  #depending on your data this might be large or small
eps = epsReal - 1e-7 # small constant to offset floating-point erros

# The network N classfies x as belonging to class 2
original_class = N(x).argmax(dim=1).item()  # TO LEARN: make sure you understand this expression
print(N(x).shape) # [1,3] as x.shape [1,10] --NN--> [10,3]. 1 is batch size, 3 is number of classes. dim=1 takes the 3 classes scores.
print("Original output: ", N(x))
print("Original Class: ", original_class)
assert(original_class == 2)

# compute gradient
# note that CrossEntropyLoss() combines the cross-entropy loss and an implicit softmax function
L = nn.CrossEntropyLoss() #-log of softmax prob the NN gives to correct class
loss = L(N(x), torch.tensor([t], dtype=torch.long)) # TO LEARN: make sure you understand this line
print("Loss: ", loss)
print(torch.softmax(N(x), dim=1))
# print(-np.log(0.3210))
loss.backward()
# your code here
# adv_x should be computed from x according to the fgsm-style perturbation such that the new class of xBar is the target class t above
# hint: you can compute the gradient of the loss w.r.t to x as x.grad
adv_x = x - eps * torch.sign(x.grad)
print("Original input: ", x)
print("Adversarial input: ", adv_x)
print("Attacked output: ", N(adv_x))
new_class = N(adv_x).argmax(dim=1).item()
print("New Class: ", new_class)

assert(new_class == t)
# it is not enough that adv_x is classified as t. We also need to make sure it is 'close' to the original x. 
print(torch.norm((x-adv_x),  p=float('inf')).data)
assert( torch.norm((x-adv_x), p=float('inf')) <= epsReal)