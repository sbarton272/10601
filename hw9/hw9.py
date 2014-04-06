import math
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
ln = math.log
sqrt = math.sqrt
pi = math.pi

# 1a
e = .4
n = (e*(1-e))/((.005/4.66)**2)
print "1.a) ", n

# 1b
n = 100
e = .4 # assume e is worst case

w = 2*2.33*math.sqrt( (e*(1-e)) / n )
print "1.b) ", w

# 1c
n = 100
e = .4
i = .005

z = i / 2 / math.sqrt( (e*(1-e)) / n )
print "1.c) ",z # 4.0%

# 2a
e1 = 1 - 86/120.0 # nn
n1 = 120
e2 = 1 - 135/200.0 # dt
n2 = 200
z = 1.96

d = e1 - e2
i = z * math.sqrt( e1*(1-e1) / n1 + e2*(1-e2) / n2 )

print "2.a) ", (d-i, d+i)

# ML.2.A.a

X = [1.3002, 10.0698, 5.4508, 3.8739, 5.4295, 3.5901, 3.7517, 6.9794, 6.8181, 
6.8344, 5.3430, 1.5850, 5.4345, 7.2605, 4.9778, 6.0694, 5.4538, 3.3931, 
4.5877, 2.4254]

lamda = len(X) / sum(X)
print "2.A.a) ", lamda

# ML.2.A.b

logLikelihood = sum([ ln(lamda) - lamda*x for x in X ])
print "2.A.b) ", logLikelihood

# ML.2.B.a

mu = np.mean(X)
print "2.B.a) ", mu

# ML.2.B.b

logLikelihood = sum([ -ln(2*sqrt(2*pi)) - (x-mu)**2/8 for x in X ])
print "2.B.b) ", logLikelihood

# ML.2.C

plt.hist(X, 5)

# ML.3.A

X = [0.89, 0.92, 0.53, 0.65, 0.99, 0.78, 0.85, 0.81, 0.31, 0.12]

alphaSqr = -len(X)/sum( [ ln(x) for x in X ] ) - 1
print "3.A) ", alphaSqr, sqrt(alphaSqr)

# ML.3.B

s = sum( [ ln(x) for x in X ] )
n = len(X)
alphaSqr = (n / (.5-s)) - 1 
print "3.B) ", alphaSqr, sqrt(alphaSqr)

# ML.4.A

p = (.015*(1-.003)) / (.015*(1-.003) + .997*.003)
print "4.A) ", p

# ML.4.B

p = (.015**2 *(1-.003)) / (.015**2 *(1-.003) + .997**2 *.003)
print "4.B) ", p

# ML.5.A

X = [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

p = sum(X[0:2]) / 2.0
print "5.A) ", p

# ML.5.B

p = sum(X[0:5]) / 5.0
print "5.B) ", p

# ML.5.C

p = sum(X[0:20]) / 20.0
print "5.C) ", p

# ML.5.D

p = sum(X) / (1.0*len(X))
print "5.D) ", p

# ML.6.A

X = [1.4664, -0.0026, 2.9642, 2.5201, 1.9800, 1.9652, 1.2018, 3.0187, 1.8668, 1.2855, 3.3514, 
	 1.7752, 1.4110, 1.7062, 1.1521, 0.8799, 4.5260, 3.6555, 2.3075, 0.7429]

mean = np.mean(X)
sumX = sum(X)
n = len(X)

def L(mu,sig):
	return (mu + sig**2*sumX) / (1 + n*sig**2)

mu = mean
print "6.A) ", mu

# ML.6.B

mu = L(1,1) 
print "6.B) ", mu

# ML.6.C

mu = L(1,2)
print "6.C) ", mu

# ML.6.D

mu = L(0,1)
print "6.D) ", mu

# end
# plt.show()