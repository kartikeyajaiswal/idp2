import scipy
import numpy as np
from matplotlib import pyplot as plt
MSS = np.load('mss.npy')
plt.imshow(MonaLisa,'gray')
#plt.show()

#rate = 0.5, n=8, k=4
img = np.reshape(MonaLisa, (2750,4))
G1 = [[1,1,1,1,0,0,0,0],[1,1,0,0,1,1,0,0],[1,0,1,0,1,0,1,0],[0,1,1,0,1,0,0,1]]
chc = np.dot(img, G1)

chc2 = np.remainder(chc, 2)
print(chc2)


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
		x[2*j-1] = -1
	else:
		x[2*j-1] = 1

for k in range(5500):
	for l in range(50):
		s[k][l] = x[2*k-1]*np.cos(2*l*np.pi*fc/fs) + x[2*k]*np.sin(2*l*np.pi*fc/fs)


n = np.random.normal(0,1.97,275000)
n = np.reshape(n, (5500,50))

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
	
	mindist[k] = min(sum1[k], sum2[k], sum3[k], sum4[k])
	
	if sum1[k] == mindist[k] :
		a[k] = [0, 0]
	elif sum2[k] == mindist[k]:
		a[k] = [0, 1]
	elif sum3[k] == mindist[k]:
		a[k] = [1, 0]
	elif sum4[k] == mindist[k]:
		a[k] = [1, 1]

pic2 = np.reshape(a, (110,100))
#plt.imshow(pic2, 'gray')
#plt.savefig('-10db.png')
#plt.show()

#bit error
biterror = np.reshape(MonaLisa - pic2, (11000,1))
print(np.count_nonzero(biterror))
BER = np.count_nonzero(biterror)*100/11000
print(BER)

'''         |standard deviation|  bit error  |
	 -10 dB	|		11.18	   |	32.38%	 |
	 -5 dB	|		06.22	   |	21.93%	 |
	  0 dB	|		03.53	   |	08.17%	 |
	  5 dB	|	    01.97      |	00.56%	 |
	 