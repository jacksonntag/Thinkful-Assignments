import numpy as np

x=0
list=[]
def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors
list = prime_factors(600851475143)
print (list)
x=max(list)
print(x)
