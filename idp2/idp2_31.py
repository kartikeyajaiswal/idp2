import scipy
from scipy.spatial import distance
import numpy as np
from matplotlib import pyplot as plt
MSS = np.load('mss.npy')
plt.imshow(MSS,'gray')
#plt.show()

#rate = 1/3, n=12, k=4
img = np.reshape(MSS, (30000,4))
G1 = [[1,0,0,0,0,1,1,1,1,0,1,0],[0,1,0,0,1,0,1,1,0,1,1,0],[0,0,1,0,1,1,1,0,1,1,1,1],[0,0,0,1,0,0,0,1,1,1,1,1]]
chc = np.dot(img, G1)
ham = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]]
ham2 = np.dot(ham, G1)
ham2 = np.remainder(ham2, 2)
chc2 = np.remainder(chc, 2)
print(chc2)
print(ham2)

#transmitting signal 


#transmitted signal
pic = np.reshape(chc2, (180000,2))
x = np.zeros(360000)
s = np.zeros((180000,50))
fc = 2000000
fs = 50000000
for i in range(180000):
	if pic[i][0] == 1:
		x[2*i] = -1
	else:
		x[2*i] = 1
for j in range(180000):
	if pic[j][1] == 1:
		x[2*j-1] = -1
	else:
		x[2*j-1] = 1

for k in range(180000):
	for l in range(50):
		s[k][l] = x[2*k-1]*np.cos(2*l*np.pi*fc/fs) + x[2*k]*np.sin(2*l*np.pi*fc/fs)


n = np.random.normal(0,6.29,9000000)
n = np.reshape(n, (180000,50))

#recieved signal
r = s + n

s1 = np.zeros(50)
s2 = np.zeros(50)
s3 = np.zeros(50)
s4 = np.zeros(50)

for l in range(50):
	s1[l] =  1*np.cos(2*l*np.pi*fc/fs) + 1*np.sin(2*l*np.pi*fc/fs)

for l in range(50):
	s2[l] = -1*np.cos(2*l*np.pi*fc/fs) + 1*np.sin(2*l*np.pi*fc/fs)

for l in range(50):
	s3[l] =  1*np.cos(2*l*np.pi*fc/fs) - 1*np.sin(2*l*np.pi*fc/fs)

for l in range(50):
	s4[l] = -1*np.cos(2*l*np.pi*fc/fs) - 1*np.sin(2*l*np.pi*fc/fs)


#demodulation
diff1 = np.zeros((180000,50))
diff2 = np.zeros((180000,50))
diff3 = np.zeros((180000,50))
diff4 = np.zeros((180000,50))

sum1 = np.zeros(180000)
sum2 = np.zeros(180000)
sum3 = np.zeros(180000)
sum4 = np.zeros(180000)

mindist = np.zeros(180000)
a = np.zeros((180000,2))

for k in range(180000):
	diff1[k] = (r[k] - s1)**2
	diff2[k] = (r[k] - s2)**2
	diff3[k] = (r[k] - s3)**2
	diff4[k] = (r[k] - s4)**2
	
	sum1[k] = sum(diff1[k])
	sum2[k] = sum(diff2[k])
	sum3[k] = sum(diff3[k])
	sum4[k] = sum(diff4[k])
	
	mindist[k] = min(sum1[k], sum2[k], sum3[k], sum4[k])
	
	if sum1[k] == mindist[k] :
		a[k] = [0, 0]
	elif sum2[k] == mindist[k]:
		a[k] = [0, 1]
	elif sum3[k] == mindist[k]:
		a[k] = [1, 0]
	elif sum4[k] == mindist[k]:
		a[k] = [1, 1]

print('mod')
y = np.reshape(a, (30000,12))
print('rec')
print(y)


demod = np.zeros((30000,4))
minh = np.zeros(30000)
#hamming distance

for z in range(30000):
	h2 = np.zeros(16)
	for z1 in range(16):
		h2[z1] = distance.hamming(y[z], ham2[z1])
		demod[z] = ham[min(np.where(h2 == np.min(h2))[0])]


demodpic = np.reshape(demod,(400,300))
plt.imshow(demodpic,'gray')
plt.show()

biterror = np.reshape(MSS - demodpic,  (120000,1))
print(np.count_nonzero(biterror))
BER = np.count_nonzero(biterror)*100/120000
print(BER)