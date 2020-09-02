import scipy
from scipy.spatial import distance
import numpy as np
from matplotlib import pyplot as plt
MSS = np.load('mss.npy')
plt.imshow(MSS,'gray')
#plt.show()

#rate = 0.5, n=8, k=4
img = np.reshape(MSS, (120000,1))
img3 = np.zeros((120000,3))
for i in range(120000):
	for j in range(3):
		img3[i][j] = img[i]
#transmitting signal 


#transmitted signal
pic = np.reshape(img3, (180000,2))
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


n = np.random.normal(0,3.069,9000000)
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
y = np.reshape(a, (120000,3))
print('rec')
print(y[5])

demod = np.zeros((120000,1))

for j in range(120000):
	if np.count_nonzero(y[j]) < 2:
		demod[j] = 0
	else:
		demod[j] = 1



demodpic = np.reshape(demod,(400,300))
plt.imshow(demodpic,'gray')
plt.savefig('6dB_2')
plt.show()


biterror = np.reshape(MSS - demodpic,  (120000,1))
print(np.count_nonzero(biterror))
BER = np.count_nonzero(biterror)*100/120000
print(BER) 
