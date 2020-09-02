import matplotlib.pyplot as plot
import numpy
import math

variance = [5, 7, 12, 20]

ber1 = [0.0485, 0.0121, 0.0008, 0.00005]

Eb_No = [-2, 0, 2, 4, 6]

ber2 = [0.2084, 0.1355, 0.07, 0.0230, 0.0041]

plot.grid(True, which = "both")


plot.semilogy(Eb_No, ber2)
plot.savefig('ebnober3')

plot.xlim([-4, 8])
plot.ylim([0.001, 0.4])
plot.title('Eb/No vs BER (channel code 3)')
plot.xlabel('Eb/No')
plot.ylabel('BER')
plot.show()