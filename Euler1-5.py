"""
Euler drills #6-10
"""
# ONE
i=1
count = 0
sum=0
while i < 1000:
    if (i % 3 is 0) or (i % 5 is 0):
        sum +=i
 
    i+=1
print("#1= ",sum)

#2  Even Fibonacci numbers
sum=0
this=0
evens=0
next = this +1

while sum < 4000000: 

    if (sum %2 == 0):
        evens+=sum
    sum = this + next
    this=next
    next=sum
    
print("#2= ",evens)

#3 Largest prime factor
x=0
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
print ("#3 = ",max(prime_factors(600851475143)))

#4 Largest palindrome product
j=i=0
end=1000
new=maxval=0
b=[]
while i < end:
    while j < end:
        new = i * j
        j=j+1
        f = str(new)
        b=f[::-1] 
        if ( f == b):
            new= int(f)
            if new > maxval:
                maxval = new
    j=1
    i=i+1
print ("#4 = ",maxval)
   
# 5 Smallest multiple
i = 20
while ((i %  2 != 0 or i %  3 != 0 or i %  4 != 0 or i %  5 != 0 or
         i %  6 != 0 or i %  7 != 0 or i %  8 != 0 or i %  9 != 0 or
         i % 10 != 0 or i % 11 != 0 or i % 12 != 0 or i % 13 != 0 or
         i % 14 != 0 or i % 15 != 0 or i % 16 != 0 or i % 17 != 0 or 
         i % 18 != 0 or i % 19 != 0 or i % 20 != 0) == True):

    i += 20
print ("{0:,.0f}".format(i))
