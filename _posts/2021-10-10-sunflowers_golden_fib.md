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

The Fiba

## Golden Ratio
The golden ratio is defined to be the following value:

$$ \varphi = \frac{1 + \sqrt{5}}{2} \approx 1.618033 $$ 

and it is inverse:

$$ \psi = -\varphi^{-1} \approx 0.6180339$$

## $z$-transforms and Difference Equations

The Fibonacci Sequence is related to the concept of difference equations and recurrence relations. 

$$f[n] = f[n-1] + f[n-2] + x[n]$$

*Proof*:
<details>
The $z$-transform of the difference equation:

$$ \frac{F(z)}{X(z)}  = \frac{1}{(1 - z^{-1} - z^{-2})} $$

$( z^{-2} + z^{-2} - 1) = (z^{-1} + \varphi)(z^{-1} + \psi)$$


</details>