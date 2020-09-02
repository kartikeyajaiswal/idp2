import control
import matplotlib.pyplot as plt
import numpy

num = [100,500]
den = [1,3,4,12,0]

G = control.tf(num,den)
w = numpy.logspace(-3,3, 5000)
control.nyquist(G, w);
plt.show()
