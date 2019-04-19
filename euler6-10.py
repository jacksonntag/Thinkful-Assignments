# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:58:04 2019
Euler drills #6-10
@author: Jack
"""
import sympy

#Euler 6
#
i=1
sum_sqs=0
end=101
tot=0
while i < end:
    sum_sqs += i**2
    tot=tot+i
    i=i+1
print("#6 = {0:,.0f}".format((tot*tot)-sum_sqs))

#What is the 10001st prime number?By listing the first six prime 
#numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13.
#
#What is the 10001st prime number?
      
i=1
end=10000
prime_cnt=0
while prime_cnt < end:
    i=i+2
    if sympy.isprime(i) == True:
        prime_cnt +=1
print("#7 = {0:,.0f}".format(i))

      #8 
input=   "73167176531330624919225119674426574742355349194934\
96983520312774506326239578318016984801869478851843\
85861560789112949495459501737958331952853208805511\
12540698747158523863050715693290963295227443043557\
66896648950445244523161731856403098711121722383113\
62229893423380308135336276614282806444486645238749\
30358907296290491560440772390713810515859307960866\
70172427121883998797908792274921901699720888093776\
65727333001053367881220235421809751254540594752243\
52584907711670556013604839586446706324415722155397\
53697817977846174064955149290862569321978468622482\
83972241375657056057490261407972968652414535100474\
82166370484403199890008895243450658541227588666881\
16427171479924442928230863465674813919123162824586\
17866458359124566529476545682848912883142607690042\
24219022671055626321111109370544217506941658960408\
07198403850962455444362981230987879927244284909188\
84580156166097919133875499200524063689912560717606\
05886116467109405077541002256983155200055935729725\
71636269561882670428252483600823257530420752963450"
      

strlen = len(input)
i=0
a=[]
size=13
index=0
maxval=0
while index < strlen:
    a = input[index:(index+size)]
    thisa=len(a)
    product=int(a[0])
    for k in range(1,thisa):
        product = product * int(a[k])
        if product > maxval:
            maxval = product
    index=index+1
print("#8 = {0:,.0f}".format(maxval))
      #
      #Euler 9
  
for a in range(1, 400):
    for b in range(1, 400):
        c = (1000 - a) - b

        if a < b < c:
            if a**2 + b**2 == c**2:
                print("#9 = {0:,.0f}".format(a*b*c))#
#
# 10 Find the sum of all the primes below two million.                      
 #               
i=3
end=20000
sum=2
def test_prime(n):
    for x in range(2,n):
        if(n % x==0):
            return False
    return True             
#
for i in range(3,end,2):
    if test_prime(i) == True:
        sum+=i
print("#10= {0:,.0f}".format(sum))                  