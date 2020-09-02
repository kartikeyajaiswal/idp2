import numpy as np 
from matplotlib import pyplot as plt
import math
MonaLisa = np.load('binary_image.npy')
plt.imshow(MonaLisa, 'gray')
plt.show()
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

print(pic[351])
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

'''print(s1)
print(s2)
print(s3)
print(s4)
'''
diff1 = np.zeros((5500,50))
diff2 = np.zeros((5500,50))
diff3 = np.zeros((5500,50))
diff4 = np.zeros((5500,50))
diff1[4001] = (r[i] - s1)**2
diff2[4001] = (r[i] - s2)**2
diff3[4001] = (r[i] - s3)**2
diff4[4001] = (r[i] - s4)**2

#demodulation
sum1 = np.zeros(5500)
sum2 = np.zeros(5500)
sum3 = np.zeros(5500)
sum4 = np.zeros(5500)
mindist = np.zeros(5500)
a = np.zeros((5500,2))
for i in range(5500):

	diff1[i] = (r[i] - s1)**2
	diff2[i] = (r[i] - s2)**2
	diff3[i] = (r[i] - s3)**2
	diff4[i] = (r[i] - s4)**2
	
	'''diff1 = diff1**2
	diff2 = diff2**2
	diff3 = diff3**2
	diff4 = diff4**2
	'''
	
	sum1[i] = sum(diff1[i])
	sum2[i] = sum(diff2[i])
	sum3[i] = sum(diff3[i])
	sum4[i] = sum(diff4[i])


	'''sum1 = 0
	sum2 = 0
	sum3 = 0
	sum4 = 0
	for i in range (50):
		sum1 = sum1 + diff1[i]**2
		sum2 = sum2 + diff2[i]**2
		sum3 = sum3 + diff3[i]**2
		sum4 = sum4 + diff4[i]**2'''
	
	mindist[i] = min(sum1[i], sum2[i], sum3[i], sum4[i])
	
	if sum1[i] == mindist[i] :
		a[i] = [0, 0]
	elif sum2[i] == mindist[i]:
		a[i] = [0, 1]
	elif sum3[i] == mindist[i]:
		a[i] = [1, 0]
	else:
		a[i] = [1, 1]

print(a)
print(diff1[4105])
print(diff2[4105])
print(diff3[4105])
print(diff4[4105])
'''print(diff1)
print(diff2)
print(diff3)
print(diff4)'''
print(sum1[4105])
print(sum2[4105])
print(sum3[4105])
print(sum4[4105])
'''demodbit = np.zeros((5500,2))
for h in range(5500):
	demodbit[i] = demod(r,i)
print(demodbit[5400])
'''
pic2 = np.reshape(a, (110,100))
plt.imshow(pic2, 'gray')
plt.show()


