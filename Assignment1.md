
# Assignment 1 - Probability, Linear Algebra, Programming, and Git

## Virginia Farley
*Netid:  vmf7*

# Probability and Statistics Theory

# 1
Let $f(x) = \begin{cases}
                0           & x < 0  \\
                \alpha x^2  & 0 \leq x \leq 2 \\
                0           & 2 < x
            \end{cases}$
            
For what value of $\alpha$ is $f(x)$ a valid probability density function?

*Note: for all assignments, write out all equations and math for all assignments using markdown and [LaTeX](https://tobi.oetiker.ch/lshort/lshort.pdf) and show all work*

**ANSWER**
$$f(x) > 0$$
$$\int_0^2 f(x)dx = \int_0^2 \alpha x^2dx = \alpha \frac{x^3}{3} \bigg|_0^2 = \alpha \frac{(2)^3}{3} - 0 = \alpha \frac{8}{3}$$
$$\int_0^2 f(x)dx = 1$$ $$\alpha \frac{8}{3}  = 1$$
$$\boldsymbol{\alpha} \mathbf{= \frac{3}{8}}$$

## 2
What is the cumulative distribution function (CDF) that corresponds to the following probability distribution function? Please state the value of the CDF for all possible values of $x$.

$f(x) = \begin{cases}
    \frac{1}{3} & 0 < x < 3 \\
    0           & \text{otherwise}
    \end{cases}$

**ANSWER**
$$F(x) = \int_0^x f(x)dx$$ $$\mathbf{F(x) =}  \begin{cases}
\mathbf{0 }&  \mathbf{x \boldsymbol{\leq} 0}  \\
\mathbf{\frac{x}{3}} & \mathbf{0 \boldsymbol{<} x \boldsymbol{<} 3} \\
\mathbf{1} & \mathbf{x \boldsymbol{\geq} 3 }
\end{cases}$$


## 3
For the probability distribution function for the random variable $X$,

$f(x) = \begin{cases}
    \frac{1}{3} & 0 < x < 3 \\
    0           & \text{otherwise}
    \end{cases}$
    
what is the (a) expected value and (b) variance of $X$. *Show all work*.

**ANSWER**

(a) Expected value:

$$E[X] = \int_{-\infty}^{\infty}xf(x)dx = \int_0^3 \frac{x}{3} dx = \frac{x^2}{6} \bigg |_0^3 = \frac{9}{6} - 0 = \frac{3}{2}$$
$$\mathbf{E[X] = \frac{3}{2}}$$

(b) Variance:

$$E[X^2] = \int_{-\infty}^{\infty}x^2f(x)dx = \int_0^3 \frac{x^2}{3} dx = \frac{x^3}{9} \bigg|_0^3 = \frac{27}{9} - 0 = 3$$
$$Var(X) = E[X^2] - E[X]^2 = 3 - (\frac{3}{2})^2 = 3 - \frac{9}{4} = \frac{3}{4}$$
$$\mathbf{Var(X) = \frac{3}{4}}$$

## 4
Consider the following table of data that provides the values of a discrete data vector $\mathbf{x}$ of samples from the random variable $X$, where each entry in $\mathbf{x}$ is given as $x_i$.

*Table 1. Dataset N=5 observations*

|        | $x_0$ | $x_1$ | $x_2$ | $x_3$ | $x_4$ |
|------  |-------|-------|-------|-------|-------|
|$\textbf{x}$| 2     | 3     | 10    | -1    | -1    |

What is the (a) mean, (b) variance, and the  of the data? 

*Show all work. Your answer should include the definition of mean, median, and variance in the context of discrete data.*

**ANSWER**

(a) Mean

The statistical mean is the mean or average of a set of data.  It is calculated as the sum of all data points in a population, divided by the total number of data points.  The mean is used in order to determine the central tendency of the data. 

$$\bar{x} = \frac{1}{N} \sum_{i=1}^n x_i = \frac{1}{5}\sum_{i=1}^5 x_i = \frac{1}{5}(2 + 3 + 10 - 1 - 1) = \frac{1}{5}\times 13 = \frac{13}{5}$$
$$\mathbf{\bar{x} = \frac{13}{5} = 2.6}$$



(b) Variance

Variance measures how far a data set is spread out, or the variability of the distribution.  It is calculated as the sum of the squares, which is the sum of the squared differences of each data point from the mean, divided by the total number of data points minus 1. Note that the denominator is N-1, not N because this these data points represent a sample, not a population. 

$$\sigma^2 = \frac{1}{N-1}\sum_{i=1}^n (x_i - \bar{x})^2 = \frac{1}{5-1}\sum_{i=1}^5 (x_i - \bar{x})^2$$

$$\sigma^2 =\frac{1}{4}\Bigg[\bigg(2 - \frac{13}{5}\bigg)^2+\bigg(3 - \frac{13}{5}\bigg)^2)+ \bigg(10 - \frac{13}{5}\bigg)^2 + \bigg(-1 - \frac{13}{5}\bigg)^2 + \bigg(-1 - \frac{13}{5}\bigg)^2\Bigg]$$
$$\mathbf{\boldsymbol{\sigma}^2 = 20.3}$$

(c) Median

The median is the middle point of a data set.  It is calculated by ordering all the data points and selecting the midde one.  If there are two middle numbers, the median is determined by taking the mean of these two numbers.


$$\mathbf{\textbf{Median} = 2}$$



## 5
Review of counting from probability theory. 

(a) How many different 7-place licence plates are possible if the first 3 places only contain letters and te last 4 only contain numbers?

(b) How many different batting orders are possible for a baseball team with 9 players?

(c) How many batting orders of 5 players are possible for a team with 9 players total?

(d) Let's assume this class has 26 students and we want to form project teams. How many unique teams of 3 are possible?

*Hint: For each problem, determine if order matters, and if it should be calculated with or without replacement.*

**ANSWER**

(a)

The first 3 places can only contain letters (26 possibilities). The last
4 places can only contain numbers (10 possibilities).

$$\prod_{i=1}^{7} = 26 \times 26 \times 26 \times 10 \times 10 \times 10 \times 10$$
$$\mathbf{\boldsymbol{\prod_{i=1}^{7}} = 175,760,000}$$


(b)

Because batting order matters in this problem, this is a permutation.
There is no replacement in this permutation because a player cannot bat
twice in an inning.

$$\text{P(n,r)} = \frac{n!}{(n-r)!}$$
$$\text{P(9, 9)} = \frac{9!}{(9-9)!} = 9\times8\times7\times6\times5\times4\times3\times2\times1$$
$$\mathbf{\textbf{P(9,9)} =362,880}$$

(c)

Because batting order matters in this problem, this is a permutation.
There is no replacement in this permutation because a player cannot bat
twice in an inning. $$\text{P(n,r)} = \frac{n!}{(n-r)!}$$
$$\text{P(9, 5)} = \frac{9!}{(9-5)!} = \frac{9!}{4!} =\frac{9\times8\times7\times6\times5\times4\times3\times2\times1}{4\times3\times2\times1} = 9\times8\times7\times6\times5$$
$$\mathbf{\textbf{P(9,5)} =15,120}$$

(d)

This is a combination because order does not matter.

$$\text{C(n,r)} =\frac{n!}{r!(n-r)!}$$
$$\text{C(26,3)} =\frac{26!}{3!(26-3)!} = frac{26!}{3!23!}$$
$$\mathbf{\textbf{C(26,3)} =2,600}$$


# Linear Algebra

## 6
**Matrix manipulations and multiplication**. Machine learning involves working with many matrices, so this exercise will provide you with the opportunity to practice those skills.

Let
$\mathbf{A} =  \begin{bmatrix}
1 & 2 & 3 \\
2 & 4 & 5 \\
3 & 5 & 6 
\end{bmatrix}$, $\mathbf{b} =  \begin{bmatrix}
-1  \\
3  \\
8  
\end{bmatrix}$, $\mathbf{c} =  \begin{bmatrix}
4  \\
-3  \\
6  
\end{bmatrix}$, and $\mathbf{I} =  \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 
\end{bmatrix}$

Compute the following or indicate that it cannot be computed:

1. $\mathbf{A}\mathbf{A}$
2. $\mathbf{A}\mathbf{A}^T$
3. $\mathbf{A}\mathbf{b}$
4. $\mathbf{A}\mathbf{b}^T$
5. $\mathbf{b}\mathbf{A}$
6. $\mathbf{b}^T\mathbf{A}$
7. $\mathbf{b}\mathbf{b}$
8. $\mathbf{b}^T\mathbf{b}$
9. $\mathbf{b}\mathbf{b}^T$
10. $\mathbf{b} + \mathbf{c}^T$
11. $\mathbf{b}^T\mathbf{b}^T$
12. $\mathbf{A}^{-1}\mathbf{b}$
13. $\mathbf{A}\circ\mathbf{A}$
14. $\mathbf{b}\circ\mathbf{c}$

*Note: The element-wise (or Hadamard) product is the product of each element in one matrix with the corresponding element in another matrix, and is represented by the symbol "$\circ$".*

**ANSWER**

1\. $$\textbf{AA} = \begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix}
\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} = \begin{bmatrix}
    14&25&31\\
    25&45&56\\
    31&56&70\\
\end{bmatrix}$$

2\. $$\textbf{A$\mathbf{A^T}$} = \begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix}
\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix}^T =  \begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix}
\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} = \begin{bmatrix}
    14&25&31\\
    25&45&56\\
    31&56&70\\
\end{bmatrix}$$

3\. $$\textbf{Ab}= \begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} \begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix} = \begin{bmatrix}
   29\\
    50\\
    60\\
\end{bmatrix}$$

4\. $$\textbf{A$\mathbf{b^T}$} =\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} \begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix}^T = \begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\ \end{bmatrix} \begin{bmatrix}
 -1&3&8
\end{bmatrix}= \textbf{not computable}$$ 

5\. $$\textbf{bA} =\begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix}\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} = \textbf{not computable}$$

6\. $$\textbf{$\mathbf{b^TA}$} =\begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix}^T \begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} = \begin{bmatrix}
 -1&3&8
\end{bmatrix}\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} = \begin{bmatrix}
29&50&60
\end{bmatrix}$$

7\. $$\textbf{bb} = \begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix}\begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix} = \textbf{not computable}$$

8\. $$\textbf{$\mathbf{b^Tb}$} = \begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix}^T\begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix} = \begin{bmatrix}
 -1&3&8
\end{bmatrix}\begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix} = 74$$

9\. $$\textbf{b$\mathbf{b^T}$} = \begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix}\begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix}^T = \begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix}\begin{bmatrix}
 -1&3&8
\end{bmatrix} =\begin{bmatrix}
    1&-3&-8\\
    -3&9&24\\
    -8&24&64\\
\end{bmatrix}$$

10\. $$\textbf{b$+\mathbf{c^T}$} = \begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix} + \begin{bmatrix}
    4\\
    -3\\
    6\\
\end{bmatrix}^T = \begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix} + \begin{bmatrix}
    4&-3&6
\end{bmatrix} = \textbf{not computable}$$

11\. $$\textbf{$\mathbf{b^Tb^T}$} = \begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix}^T\begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix}^T = \begin{bmatrix}
 -1&3&8
\end{bmatrix}\begin{bmatrix}
 -1&3&8
\end{bmatrix} = \textbf{not computable}$$

12\. $$AA^{-1} = I$$ $$AI = 
 \left[\begin{array}{rrr|rrr}
  1 & 2 & 3 & 1 & 0 &0 \\
   2 & 4 & 5 & 0 & 1& 0\\
   3 & 5 & 6 & 0 & 0 & 1\\ 
   \end{array} \right] =  \left[\begin{array}{rrr|rrr}
  3 & 5 & 6 & 0 & 0 &1 \\
   0 & \frac{2}{3} & 1 & 0 & 1& -\frac{2}{3}\\
   0 & 0 & \frac{1}{2} & 1 & -\frac{1}{2} & 0\\ 
   \end{array} \right] = \left[\begin{array}{rrr|rrr}
  1 & 0 & 0 & 1 & -3 &2 \\
   0 & 1 & 0 & -3& 3& -1\\
   0 & 0 & 1 & 2 & -1 & 0\\ 
   \end{array} \right] = IA^{-1}$$

$$A^{-1} =\begin{bmatrix}
    1&-3&2\\
    -3&3&-1\\
    2&-1&0\\
\end{bmatrix}$$

$$\textbf{$\mathbf{A^{-1}b}$} =\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix}^{-1} \begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix}=  \begin{bmatrix}
    1&-3&2\\
    -3&3&-1\\
    2&-1&0\\
\end{bmatrix}\begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix}=\begin{bmatrix}
    6\\
    4\\
    -5\\
\end{bmatrix}$$
13\. $$\mathbf{A}\circ\mathbf{A} = \begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} \circ \begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} = \begin{bmatrix}
    1&4&9\\
    4&16&25\\
    9&25&36\\
\end{bmatrix}$$
14\. $$\mathbf{b}\circ\mathbf{c} = \begin{bmatrix}
    -1\\
    3\\
    8\\
\end{bmatrix} \circ \begin{bmatrix}
    4\\
    -3\\
    6\\
\end{bmatrix} = \begin{bmatrix}
    -4\\
    -9\\
    48\\
\end{bmatrix}$$



## 6
**Eigenvectors and eigenvalues**. Eigenvectors and eigenvalues are useful for some machine learning algorithms, but the concepts take time to solidly grasp. For an intuitive review of these concepts, explore this [interactive website at Setosa.io](http://setosa.io/ev/eigenvectors-and-eigenvalues/). Also, the series of linear algebra videos by Grant Sanderson of 3Brown1Blue are excellent and can be viewed on youtube [here](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).

1. Calculate the eigenvalues and corresponding eigenvectors of matrix $\mathbf{A}$ above, from the last question.
2. Choose one of the eigenvector/eigenvalue pairs, $\mathbf{v}$ and $\lambda$, and show that $\mathbf{A} \mathbf{v} = \lambda \mathbf{v}$. Also show that this relationship extends to higher orders: $\mathbf{A} \mathbf{A} \mathbf{v} = \lambda^2 \mathbf{v}$
3. Show that the eigenvectors are orthogonal to one another (e.g. their inner product is zero). This is true for real, symmetric matrices.


**ANSWER**

1.

$$det(A - \lambda I) = 0$$ $$det \Bigg(\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} - \lambda \begin{bmatrix}
    1&0&0\\
    0&1&0\\
    0&0&1\\
\end{bmatrix}\Bigg) = 0$$
$$-\lambda ^3 + 11 \lambda ^2 + 4\lambda - 1 = 0$$
$$\mathbf{\boldsymbol{\lambda}_1 =.1710, \boldsymbol{\lambda}_2 = -.5157,\boldsymbol{\lambda}_3 = 11.3448}$$

$$Av = \lambda v$$ $$(A - \lambda I) v = 0$$
$$A - \lambda_1 I =\Bigg(\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} - .1710 \begin{bmatrix}
    1&0&0\\
    0&1&0\\
    0&0&1\\
\end{bmatrix}\Bigg)v_1 = 0$$



$$A - \lambda_2 I =\Bigg(\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} + .5157 \begin{bmatrix}
    1&0&0\\
    0&1&0\\
    0&0&1\\
\end{bmatrix}\Bigg)v_2 = 0$$



$$A - \lambda_3 I =\Bigg(\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix} - 11.3448 \begin{bmatrix}
    1&0&0\\
    0&1&0\\
    0&0&1\\
\end{bmatrix}\Bigg)v_3 = 0$$

$$\mathbf{v_1 = \begin{bmatrix}
    \mathbf{1.8020}\\
    \mathbf{-2.2470}\\
    \mathbf{1}\\
\end{bmatrix}}, \mathbf{v_2 = \begin{bmatrix}
    \mathbf{-1.2470}\\
   \mathbf{-0.5550}\\
    \mathbf{1}\\
\end{bmatrix}}, \mathbf{v_3 = \begin{bmatrix}
    \mathbf{0.4450}\\
   \mathbf{0.8019}\\
    \mathbf{1}\\
\end{bmatrix}}$$

2.

$$Av_1 = \lambda_1 v_1$$
$$\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix}\begin{bmatrix}
    1.8020\\
    -2.2470\\
    1\\
\end{bmatrix} = .1710 \begin{bmatrix}
    1.8020\\
    -2.2470\\
    1\\
\end{bmatrix}$$
$$\begin{bmatrix}
    0.308\\
    -0.384\\
    0.171\\
\end{bmatrix} =\begin{bmatrix}
    0.308\\
    -0.384\\
    0.171\\
\end{bmatrix} $$

$$AAv_1 = \lambda_1 ^2 v$$
$$\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix}\begin{bmatrix}
    1&2&3\\
    2&4&5\\
    3&5&6\\
\end{bmatrix}\begin{bmatrix}
    1.8020\\
    -2.2470\\
    1\\
\end{bmatrix} = .1710^2 \begin{bmatrix}
    1.8020\\
    -2.2470\\
    1\\
\end{bmatrix}$$
$$\begin{bmatrix}
    0.053\\
    -0.065\\
    .03\\
\end{bmatrix} =\begin{bmatrix}
    .053\\
    -0.065\\
    .03\\
\end{bmatrix} $$

3.

For the eigenvectors to be orthogonal to each other, the dot product of every combination of two eignevectors is zero.  Because these three dot products are all zero, the eigenvectors are orthogonal.  Therefore, matrix $\mathbf{A}$ is a real, symmetric matrix.

$$ \begin{bmatrix}
    1.8020\\
    -2.2470\\
    1\\
\end{bmatrix} \circ \begin{bmatrix}
    -1.2470\\
   -0.5550\\
    1\\
\end{bmatrix} = 0$$




$$\begin{bmatrix}
    -1.2470\\
   -0.5550\\
    1\\
\end{bmatrix} \circ \begin{bmatrix}
    0.4450\\
   0.8019\\
    1\\
\end{bmatrix} = 0$$

$$\begin{bmatrix}
    1.8020\\
    -2.2470\\
    1\\
\end{bmatrix} \circ \begin{bmatrix}
    0.4450\\
   0.8019\\
    1\\
\end{bmatrix} = 0$$


# Numerical Programming

## 7
Speed comparison between vectorized and non-vectorized code. Begin by creating an array of 10 million random numbers using the numpy random.randn module. Compute the sum of the squares first in a for loop, then using Numpy's `dot` module. Time how long it takes to compute each and report the results and report the output. How many times faster is the vectorized code than the for loop approach?

*Note: all code should be well commented, properly formatted, and your answers should be output using the `print()` function as follows (where the # represents your answers, to a reasonable precision):

`Time [sec] (non-vectorized): ######`

`Time [sec] (vectorized):     ######`

`The vectorized code is ##### times faster than the vectorized code`

**ANSWER**


```python
import numpy as np
import time

# Generate the random samples
x = np.random.randn(10000, 1000)

# Compute the sum of squares the non-vectorized way (using a for loop)
sum1=0
start1 = time.time()
for i, row in enumerate(x):
    for j, col in enumerate(row):
        sum1 += x[i][j] * x[i][j]        
end1 = time.time()

t1 = end1 - start1

# Compute the sum of squares the vectorized way (using numpy)
start2 = time.time()
sum2 = np.sum(np.square(x))
end2 = time.time()

t2 = end2 - start2

# Print the results
print('Time [sec] (non-vectorized): ' + '%.4f'% t1)
print('Sum of Squares (non-vectorized): ' + '%.4f'% sum1)

print('Time [sec] (vectorized): ' + '%.4f'% t2)
print('Sum of Squares (vectorized): ' + '%.4f'% sum2)

z = t1/t2
print('The vectorized code is ' + '%.4f'% z + ' times faster than the non-vectorized code')

```

    Time [sec] (non-vectorized): 13.4708
    Sum of Squares (non-vectorized): 9997832.0580
    Time [sec] (vectorized): 0.0394
    Sum of Squares (vectorized): 9997832.0580
    The vectorized code is 341.6983 times faster than the non-vectorized code


## 8
One popular Agile development framework is Scrum (a paradigm recommended for data science projects). It emphasizes the continual evolution of code for projects, becoming progressively better, but starting with a quickly developed minimum viable product. This often means that code written early on is not optimized, and that's a good thing - it's best to get it to work first before optimizing. Imagine that you wrote the following code during a sprint towards getting an end-to-end system working. Vectorize the following code and show the difference in speed between the current implementation and a vectorized version.

The function below computes the function $f(x,y) = x^2 - 2 y^2$ and determines whether this quantity is above or below a given threshold, `thresh=0`. This is done for $x,y \in \{-4,4\}$, with 2,000 samples in each direction.

(a) Vectorize this code and demonstrate (as in the last exercise) the speed increase through vectorization and (b) plot the resulting data - both the function $f(x,y)$ and the thresholded output - using [`imshow`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html?highlight=matplotlib%20pyplot%20imshow#matplotlib.pyplot.imshow) from `matplotlib`.

*Hint: look at the `numpy` [`meshgrid`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.meshgrid.html) documentation*


```python
import numpy as np
import time
import matplotlib.pyplot as plt

# Initialize variables for this exerise
x = np.linspace(-4, 4, 2000)
y = np.linspace(-4, 4, 2000)
xx, yy = np.meshgrid(x, y, indexing = 'ij')
thresh = 0

# Nonvectorized implementation  
f1 = np.zeros(xx.shape)
f1_thresh = np.zeros(xx.shape)
start1 = time.time()
for i, row in enumerate(f1):
    for j, col in enumerate(row):
        f1[i,j] = xx[i, j] ** 2 - 2 * yy[i, j] ** 2
        if f1[i,j] >= 0:
            f1_thresh[i, j] = 1
        elif f1[i,j] < 0:
            f1_thresh[i, j] = 0
end1 = time.time()

t1 = end1 - start1

# Vectorized implementation
start2 = time.time()
f2 = np.square(xx) - 2 * np.square(yy)
f2_thresh = np.copy(f2)
f2_thresh[f2_thresh >= 0] = 1
f2_thresh[f2_thresh < 0] = 0
end2 = time.time()

t2 = end2 - start2

# Print the time for each and the speed increase
print('Time [sec] (non-vectorized): ' + '%.4f'% t1)
print('Time [sec] (vectorized): ' + '%.4f'% t2)
z = t1/t2
print('The vectorized code is ' + '%.4f'% z + ' times faster than the non-vectorized code')

# Plot the result
fig = plt.figure(1)
plt.subplot(1, 2, 1)
plt.subplot(1, 2, 1).set_title("Function F(x,y) Output")
plt.imshow(f2)
plt.subplot(1, 2, 2)
plt.subplot(1, 2, 2).set_title("Thresholded Output")
plt.imshow(f2_thresh)
plt.tight_layout()
plt.show()

```

    Time [sec] (non-vectorized): 15.4023
    Time [sec] (vectorized): 0.0969
    The vectorized code is 158.9034 times faster than the non-vectorized code



![png](output_22_1.png)


## 9
This exercise will walk through some basic numerical programming exercises.
1. Synthesize $n=10^4$ normally distributed data points with mean $\mu=2$ and a standard deviation of $\sigma=1$. Call these observations from a random variable $X$, and call the vector of observations that you generate, $\textbf{x}$.
2. Calculate the mean and standard deviation of $\textbf{x}$ to validate (1) and provide the result to a precision of four significant figures.
3. Plot a histogram of the data in $\textbf{x}$ with 30 bins
4. What is the 90th percentile of $\textbf{x}$? The 90th percentile is the value below which 90% of observations can be found.
5. What is the 99th percentile of $\textbf{x}$?
6. Now synthesize $n=10^4$ normally distributed data points with mean $\mu=0$ and a standard deviation of $\sigma=3$. Call these observations from a random variable $Y$, and call the vector of observations that you generate, $\textbf{y}$.
7. Plot the histogram of the data in $\textbf{y}$ on a (new) plot with the histogram of $\textbf{x}$, so that both histograms can be seen and compared.
8. Using the observations from $\textbf{x}$ and $\textbf{y}$, estimate $E[XY]$

**ANSWER**


```python
import numpy as np
from matplotlib import pyplot as plt

# 1
mu_x, sigma_x = 2, 1

x = []
for i in range(0, 1000):
    X = np.random.normal(mu_x, sigma_x)
    x.append(X)

# 2
mean = np.mean(x)
stdev = np.std(x)
print('The mean of x is ' + '%.4f'% mean + ' and the standard deviation of x is '  + '%.4f'% stdev + '.')

# 3
plt.hist(x, bins = 30)
plt.title("Histogram of Random Variable X with 30 Bins")
plt.show()

# 4
x_90 = np.percentile(x, 90)
print('The 90th percentile of x is ' + '%.4f'% x_90 + '.')

# 5
x_99 = np.percentile(x, 99)
print('The 99th percentile of x is ' + '%.4f'% x_99 + '.')

# 6
mu_y, sigma_y = 0, 3

y = []
for i in range(0, 1000):
    Y = np.random.normal(mu_y, sigma_y)
    y.append(Y)

# 7
plt.hist(y, bins = 30)
plt.title("Histogram of Random Variable Y with 30 Bins")
plt.show()

# 8
#E[XY] = E[X]E[Y]
expected_val = np.mean(x) * np.mean(y)
print('The estimate for E[XY] is ' + '%.4f'% expected_val + '.')
```

    The mean of x is 2.0190 and the standard deviation of x is 1.0018.



![png](output_25_1.png)


    The 90th percentile of x is 3.3155.
    The 99th percentile of x is 4.3805.



![png](output_25_3.png)


    The estimate for E[XY] is 0.1244.


## 10
Estimate the integral of the function $f(x)$ on the interval $0\leq x < 2.5$ assuming we only know the following points from $f$:

*Table 1. Dataset containing n=5 observations*

| $x_i$ | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 |
|-|-|-|-|-|-|
| $y_i$ | 6 | 7 | 8 | 4 | 1 |


**ANSWER**

$$F(x) = \int f(x) \approx \frac{2}{5}\times 6 + \frac{2}{5} \times 7 + \frac{2}{5} \times 8 + \frac{2}{5} \times 4 + \frac{2}{5} \times 1 = 2.4 + 2.8 + 3.2 + 1.6 + .4 = 10.4 $$



# Version Control via Git

## 11
Complete the [Atlassian Git tutorial](https://www.atlassian.com/git/tutorials/what-is-version-control), specifically the following sections. Try each concept that's presented. For this tutorial, instead of using BitBucket, use Github. Create a github account here if you don't already have one: https://github.com/
1. [What is version control](https://www.atlassian.com/git/tutorials/what-is-version-control)
2. [What is Git](https://www.atlassian.com/git/tutorials/what-is-git)
3. [Install Git](https://www.atlassian.com/git/tutorials/install-git)
4. [Setting up a repository](https://www.atlassian.com/git/tutorials/install-git)
5. [Saving changes](https://www.atlassian.com/git/tutorials/saving-changes)
6. [Inspecting a repository](https://www.atlassian.com/git/tutorials/inspecting-a-repository)
7. [Undoing changes](https://www.atlassian.com/git/tutorials/undoing-changes)
8. [Rewriting history](https://www.atlassian.com/git/tutorials/rewriting-history)
9. [Syncing](https://www.atlassian.com/git/tutorials/syncing)
10. [Making a pull request](https://www.atlassian.com/git/tutorials/making-a-pull-request)
11. [Using branches](https://www.atlassian.com/git/tutorials/using-branches)
12. [Comparing workflows](https://www.atlassian.com/git/tutorials/comparing-workflows)

For your answer, affirm that you either completed the tutorial or have previous experience with all of the concepts above. Do this by typing your name below and selecting the situation that applies from the two options in brackets.

**ANSWER**

*I, Virginia Farley, affirm that I have completed the above tutorial.*

## 12
Using Github to create a static HTML website:
1. Create a branch in your `machine-learning-course` repo called "gh-pages" and checkout that branch (this will provide an example of how to create a simple static website using [Github Pages](https://pages.github.com/))
2. Create a file called "index.html" with the contents "Hello World" and add, commit, and push it to that branch.
3. Submit the following: (a) a link to your github repository and (b) a link to your new "Hello World" website. The latter should be at the address https://[USERNAME].github.io/ECE590-assignment0 (where [USERNAME] is your github username).

**ANSWER**  
(a) Link to GitHub repository: [https://github.com/vmf7/machine-learning-course](https://github.com/vmf7/machine-learning-course)  
(b) Link to "Hello World" website: [https://vmf7.github.io/ECE590-assignment0](https://vmf7.github.io/ECE590-assignment0)



# Exploratory Data Analysis
## 13
Here you'll bring together some of the individual skills that you demonstrated above and create a Jupyter notebook based blog post on data analysis.

1. Find a dataset that interests you and relates to a question or problem that you find intriguing
2. Using a Jupyter notebook, describe the dataset, the source of the data, and the reason the dataset was of interest.
3. Check the data and see if they need to be cleaned: are there missing values? Are there clearly erroneous values? Do two tables need to be merged together? Clean the data so it can be visualized.
3. Plot the data, demonstrating interesting features that you discover. Are there any relationships between variables that were surprising or patterns that emerged? Please exercise creativity and curiosity in your plots.
4. What insights are you able to take away from exploring the data? Is there a reason why analyzing the dataset you chose is particularly interesting or important? Summarize this as if your target audience was the readership of a major news organization - boil down your findings in a way that is accessible, but still accurate.
5. Create a public repository on your github account titled "machine-learning-course". In it, create a readme file that contains the heading "ECE590: Introductory Machine Learning for Data Science". Add, commit, and push that Jupyter notebook to the master branch. Provide the link to the that post here.

**ANSWER**  
Link to blog post on data analysis: [https://github.com/vmf7/machine-learning-course/blob/master/Exploratory_Data_Analysis.md](https://github.com/vmf7/machine-learning-course/blob/master/Exploratory_Data_Analysis.md)


