# Sunflowers, Fibonacci, Golden Ratio

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



## Golden Ratio
The golden ratio is defined to be the following value:

$$ \varphi = \frac{1 + \sqrt{5}}{2} \approx 1.618033 $$


The Fibonacci Sequence can be expressed in a closed form function that includes the golden ratio. 

### $z$-transforms and Recurrence relations

$z$-transforms are useful

<details>
$$F(z) = z^{-1}F(z) + z^{-2} F(z) + X(z)$$

$$ \frac{F(z)}{X(z)} = \frac{-1}{(z^{-2} + z^{-1} - 1)} $$

$z^{-2} + z^{-1} - 1$ factorizes into $(z^{-1} + \varphi)(z^{-1} - \varphi^{-1} )$ (using the quadratic formula)


<p>Using a partial fraction decomposition</p>

$$ \frac{-1}{(z^{-2} + z^{-1} - 1)} = \frac{A}{(z^{-1} + \varphi)} + \frac{B}{(z^{-1} - \varphi^{-1} )} $$

$$ 

A + B = 0 \\
-\varphi^{-1} A + \varphi B = -1
$$

$$ A = \frac{-1}{\varphi - \varphi^{-1} } \; B = \frac{1}{\varphi - \varphi^{-1} }$$


$$\frac{A\varphi^{-1}}{(\varphi^{-1}z^{-1} + 1)} + \frac{-\varphi B }{(- \varphi z^{-1} + 1)}$$

Using the inverse $z$-transform

$$\frac{A\varphi^{-1}}{(\varphi^{-1}z^{-1} + 1)} + \frac{-\varphi B }{(- \varphi z^{-1} + 1)}$$

</details>