import numpy as np 
from matplotlib import pyplot as plt
import math
MonaLisa = np.load('binary_image.npy')
plt.imshow(MonaLisa, 'gray')
#plt.show()
#transmitted signal
pic = np.reshape(MonaLisa, (5500,2))
x = np.zeros(11000)
s = np.zeros((5500,50))
fc = 2000000
fs = 50000000
for i in range(5500):
	if pic[i][0] == 1:
		x[2*i] = -1
	else:
		x[2*i] = 1
for j in range(5500):
	if pic[j][1] == 1:
		x[2*j - 1] = -1
	else:
		x[2*j - 1] = 1

for k in range(5500):
	for l in range(50):
		s[k][l] = x[2*k - 1]*np.cos(2*l*np.pi*fc/fs) + x[2*k]*np.sin(2*l*np.pi*fc/fs)


n = np.random.normal(0,0.1,275000)
n = np.reshape(n, (5500,50))

#recieved signal
r = s 

s1 = np.zeros(50)
s2 = np.zeros(50)
s3 = np.zeros(50)
s4 = np.zeros(50)

for l in range(50):
	s1[l] = 1*np.cos(2*l*np.pi*fc/fs) + 1*np.sin(2*l*np.pi*fc/fs)

for l in range(50):
	s2[l] = 1*np.cos(2*l*np.pi*fc/fs) - 1*np.sin(2*l*np.pi*fc/fs)

for l in range(50):
	s3[l] = -1*np.cos(2*l*np.pi*fc/fs) + 1*np.sin(2*l*np.pi*fc/fs)

for l in range(50):
	s4[l] = -1*np.cos(2*l*np.pi*fc/fs) - 1*np.sin(2*l*np.pi*fc/fs)


#demodulation
diff1 = np.zeros((5500,50))
diff2 = np.zeros((5500,50))
diff3 = np.zeros((5500,50))
diff4 = np.zeros((5500,50))

sum1 = np.zeros(5500)
sum2 = np.zeros(5500)
sum3 = np.zeros(5500)
sum4 = np.zeros(5500)

mindist = np.zeros(5500)
a = np.zeros((5500,2))

for k in range(5500):
	diff1[k] = (r[k] - s1)**2
	diff2[k] = (r[k] - s2)**2
	diff3[k] = (r[k] - s3)**2
	diff4[k] = (r[k] - s4)**2
	
	sum1[k] = sum(diff1[k])
	sum2[k] = sum(diff2[k])
	sum3[k] = sum(diff3[k])
	sum4[k] = sum(diff4[k])


	'''sum1 = 0
	sum2 = 0
	sum3 = 0
	sum4 = 0
	for i in range (50):
		sum1 = sum1 + diff1[i]**2
		sum2 = sum2 + diff2[i]**2
		sum3 = sum3 + diff3[i]**2
		sum4 = sum4 + diff4[i]**2
	'''
	mindist[k] = min(sum1[k], sum2[k], sum3[k], sum4[k])
	
	if sum1[k] == mindist[k] :
		a[k] = [0, 0]
	elif sum2[k] == mindist[k]:
		a[k] = [1, 0]
	elif sum3[k] == mindist[k]:
		a[k] = [0, 1]
	elif sum4[k] == mindist[k]:
		a[k] = [1, 1]

	

'''demodbit = np.zeros((5500,2))
for h in range(5500):
	demodbit[i] = demod(r,i)
print(demodbit[5400])
'''

pic2 = np.reshape(a, (110,100))
plt.imshow(pic2, 'gray')
plt.show()
