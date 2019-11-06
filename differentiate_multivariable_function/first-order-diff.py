import chainer
from chainer import Variable
import numpy as np

# TODO: should be made from input or file through parser
x = Variable(np.array([1,2], dtype = np.float32))
f = 2 * x[0] ** 2 + x[1] ** 2 + 2 * x[0] + x[1]
parsed_f = "2 * x[0] ** 2 + x[1] ** 2 + 2 * x[0] + x[1]"

print("I will compute the gradient of a multivaribale function f!\n")
print("f = {}\n".format(parsed_f))

print("value at the {} of f = {}\n".format(x, f))

f.grad = np.ones_like(f.array, dtype=np.float32)
f.backward()

print("gradient of f at ({})= {}".format(x, x.grad))

print("to compute gradient of intermediate Variable\n")
print("we need to call backward method with retain_grad=True\n")



# obtain gradient of intermediate variable
x = Variable(np.array([1], dtype=np.float32))
y = x **2
z = y **2
z.backward()
print("problem setting\nx:variable(for example x=1)\ny=x**2(intermediate varible)\nz=y**2")
print("if you call backward method with nothing (z.backward()))\n")
print("you can't get gradient of intermediate variable\n")
print("y = {}".format(y.grad))

print("if you call backward method with retain_grad (z.backward(retain_grad=True))\n")

x = Variable(np.array([1], dtype=np.float32))
y = x **2
z = y **2
z.backward(retain_grad=True)

print("y = {}".format(y.grad))

