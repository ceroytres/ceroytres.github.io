# Sunflowers, Fibonacci, Golden Ratio, Spirals

__in progress__

## Fibonacci Sequence

The Fibonacci Sequence is defined as the following sequence:

$$ F_0 = 0 \; F_1 = 1 $$

$$ F_{n} = F_{n-1} + F_{n-2} \; \text{ for } n > 1$$



```python
def fib(n):
    """ Recursive Fibonacci """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)
```

The Fibonacci Sequence can be expressed in closed-form 

## Golden Ratio
The golden ratio is defined to be the following value:

$$ \varphi = \frac{1 + \sqrt{5}}{2} \approx 1.618033 $$ 

and it is inverse:

$$ \psi = -\varphi^{-1} = - \approx 0.6180339$$

## $z$-transforms and Recurrence Relations

The Fibonacci Sequence is related to the concept of recurrence relations. 

$$f_n = f_{n-1} + f_{n-2}$$

*Proof*:
<details>



</details>