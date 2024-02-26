#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/drive/1TPlDVlWCwfXB0lbTJLNnHCoFwIjVgoNH?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # NumPy

# In[211]:


import numpy as np
from IPython.display import Latex


# `NumPy` is a popular `Python` library for data science focusing on arrays, vectors, and matrices. It is a fundamental library for scientific computing, and it is built on top of the `C` programming language. It provides a high-performance multidimensional array object and tools for working with these arrays. Next example will show how fast `NumPy` is compared to the built-in `Python` lists.

# In[65]:


get_ipython().run_line_magic('timeit', '[i**2 for i in range(10000)]')
get_ipython().run_line_magic('timeit', 'np.arange(10000)**2')


# ## Arrays

# ### **One-dimensional arrays**

# In[66]:


np.array([1, 2, 3, 4, 5])


# In[67]:


np.array([1, 2, 3, 4, 5], dtype='float32')


# All supported data types in `NumPy` are listed [here](https://numpy.org/doc/stable/user/basics.types.html).

# In[68]:


np.arange(1, 6), np.arange(1, 6, dtype='float32')


# In[69]:


np.arange(10, 30, 5)


# In[70]:


np.linspace(10, 30, 5)


# ### **Multidimensional arrays**
# 
# The `ndarray` is the main class of `NumPy`. It is a multidimensional array of elements of the same type. It is a table of elements (usually numbers), all of the same type, indexed by a tuple of positive integers. In `NumPy` dimensions are called axes. The number of axes is rank.

# In[71]:


matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
matrix, matrix.shape, matrix.ndim, matrix.size, matrix.dtype, type(matrix)


# In[14]:


np.arange(-2, 13).reshape(3, 5)


# In[73]:


np.zeros((3, 3)), np.ones((3, 3)), np.eye(3), np.diag(np.arange(1, 4), k=1), np.random.rand(3, 3)


# ### **Indexing and slicing**

# <img src="https://github.com/waterhackweek/learning-resources/blob/master/notebooks/img/numpy_indexing.png?raw=1" width="600"/>

# In[7]:


arr = np.arange(1, 28).reshape(3, 3, 3)
arr


# 

# In[8]:


arr[0]


# In[76]:


arr[:, 1]


# In[77]:


arr[::2, 1::2]


# In[78]:


arr[0, 0, [0, -1]]


# In[79]:


# Fancy indexing
arr[[0, 1, 2], [0, 1, 2]]


# ### **Masking**

# In[80]:


mask = arr > 14
mask


# In[81]:


arr[mask]


# In[82]:


mask1 = (arr % 6 == 0) | (arr % 4 == 0)
mask2 = np.logical_or(arr % 6 == 0, arr % 4 == 0)
arr[mask1], arr[mask2]


# In[83]:


arr[mask1] = -1
arr


# ### **Useful methods**

# All the methods of the `ndarray` class can be found [here](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html).
# 
# If you forget what a method does, you can use **?** to get the documentation.

# In[84]:


arr = np.arange(1, 28).reshape(3, 3, 3)

for func in (arr.max,
             arr.min,
             arr.mean,
             arr.sum,
             arr.std,
             arr.var,
             arr.prod,
             arr.cumsum
             ):
    print(func.__name__, "=", func())


# In[85]:


get_ipython().run_line_magic('pinfo', 'np.ndarray.flatten')


# Many methods have an `axis` parameter. If you set `axis=0`, the method will be applied to each column, if you set `axis=1`, the method will be applied to each row and so on.

# In[86]:


print(arr)
print(arr.sum())
print(arr.sum(axis=0))
print(arr.sum(axis=1))


# Sorting arrays: `np.sort()`, `np.argsort()`, `np.argmax()`, `np.argmin()`

# In[87]:


arr = np.random.randn(5)
print(arr)
print(np.sort(arr), np.argsort(arr))


# In[88]:


a = np.floor(10 * np.random.rand(2, 2))
print('Array a\n', a)

b = np.floor(10 * np.random.rand(2, 2))
print('Array b\n', b)

print('\n---Concatenate a and b vertically---\n')
print(np.vstack((a, b)))
print('\n---Concatenate a and b horizontally---\n')
print(np.hstack((a, b)))


# ### **Array broadcasting**
# The term broadcasting describes how `NumPy` treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes. Broadcasting provides a means of vectorizing array operations so that looping occurs in `C` instead of `Python`. It does this without making needless copies of data and usually leads to efficient algorithm implementations.

# <img src="http://scipy-lectures.org/_images/numpy_broadcasting.png" width="600"/>

# In[89]:


x = np.ones((3, 4))
y = np.arange(4)
print(x)
print(y)
# Add `x` and `y`. Note that `x` and `y` have different shapes.
print(x.shape, y.shape)
print(x + y)


# In[90]:


x = np.ones((3, 4))
y = np.arange(3)

print(x)
print(y)


# How to add a vector to each row of a matrix?
# The `x+y` provides an `ValueError` because the shapes of the arrays are different.

# In[91]:


# Answer


# # Matplotlib

# <img src="https://res.cloudinary.com/codecrucks/image/upload/c_scale,w_700,h_327,dpr_2/f_webp,q_auto/v1648958444/chart-types.png?_i=AA" width="800"/>

# <img src="http://blog.atkcg.ru/wp-content/uploads/2015/07/1-%D0%B9-%D1%81%D0%BB%D0%B0%D0%B9%D0%B41.jpg" width="800"/>

# Here is an image illustrating the anatomy of a `Matplotlib` plot. 
# 
# <img src="https://blog.logrocket.com/wp-content/uploads/2021/11/anatomy-figure.png" width="800"/>
# 
# `Matplotlib` is a comprehensive library for creating static, animated, and interactive visualizations in `Python`. It is a multi-platform data visualization library built on `NumPy` arrays and designed to work with the broader `SciPy` stack. It was introduced by John Hunter in 2003 and is now maintained by a large team of developers. It is a great library for making publication-quality plots.

# ## Plotting your first graph

# First we need to import the `matplotlib` library.

# In[15]:


import matplotlib


# Matplotlib can output graphs using various backend graphics libraries, such as Tk, wxPython, etc.  When running python using the command line, the graphs are typically shown in a separate window. In a Jupyter notebook, we can simply output the graphs within the notebook itself by running the `%matplotlib inline` magic command.

# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')


# Now let's plot our first graph! :)

# In[17]:


import matplotlib.pyplot as plt
plt.plot([1, 2, 4, 9, 5, 3])
plt.show()


# Yep, it's as simple as calling the `plot` function with some data, and then calling the `show` function!
# 
# If the `plot` function is given one array of data, it will use it as the coordinates on the vertical axis, and it will just use each data point's index in the array as the horizontal coordinate.
# You can also provide two arrays: one for the horizontal axis `x`, and the second for the vertical axis `y`:

# In[95]:


plt.plot([-3, -2, 5, 0], [1, 6, 4, 3])
plt.show()


# The axes automatically match the extent of the data.  We would like to give the graph a bit more room, so let's call the `axis` function to change the extent of each axis `[xmin, xmax, ymin, ymax]`.

# In[96]:


plt.plot([-3, -2, 5, 0], [1, 6, 4, 3])
plt.axis([-4, 6, 0, 7])
plt.show()


# Now, let's plot a mathematical function. We use NumPy's `linspace` function to create an array `x` containing 500 floats ranging from -2 to 2, then we create a second array `y` computed as the square of `x`.

# In[97]:


x = np.linspace(-2, 2, 500)
y = x**2

plt.plot(x, y)
plt.show()


# That's a bit dry, let's add a title, and x and y labels, and draw a grid.

# In[98]:


plt.plot(x, y)
plt.title('Square function')
plt.xlabel('x')
plt.ylabel(f'y = $x^2$')
plt.grid(True)
plt.show()


# ## Line style and color

# By default, matplotlib draws a line between consecutive points.

# In[99]:


plt.plot([0, 100, 100, 0, 0, 100, 50, 0, 100], [0, 0, 100, 100, 0, 100, 130, 100, 0])
plt.axis([-10, 110, -10, 140])
plt.show()


# You can pass a 3rd argument to change the line's style and color.
# For example `"g--"` means "green dashed line".

# In[100]:


plt.plot([0, 100, 100, 0, 0, 100, 50, 0, 100], [0, 0, 100, 100, 0, 100, 130, 100, 0], "g--")
plt.axis([-10, 110, -10, 140])
plt.show()


# You can plot multiple lines on one graph very simply: just pass `x1, y1, [style1], x2, y2, [style2], ...`
# 
# For example:

# In[101]:


plt.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], "r-", [0, 100, 50, 0, 100], [0, 100, 130, 100, 0], "g--")
plt.axis([-10, 110, -10, 140])
plt.show()


# Or simply call `plot` multiple times before calling `show`.

# In[102]:


plt.plot([0, 100, 100, 0, 0], [0, 0, 100, 100, 0], "r-")
plt.plot([0, 100, 50, 0, 100], [0, 100, 130, 100, 0], "g--")
plt.axis([-10, 110, -10, 140])
plt.show()


# You can also draw simple points instead of lines. Here's an example with green dashes, red dotted line and blue triangles.
# Check out [the documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot) for the full list of style & color options.

# In[103]:


x = np.linspace(-1.4, 1.4, 30)
plt.plot(x, x, 'g--', x, x**2, 'r:', x, x**3, 'b^')
plt.show()


# The plot function returns a list of `Line2D` objects (one for each line).  You can set extra attributes on these lines, such as the line width, the dash style or the alpha level.  See the full list of attributes in [the documentation](http://matplotlib.org/users/pyplot_tutorial.html#controlling-line-properties).

# In[104]:


x = np.linspace(-1.4, 1.4, 30)
line1, line2, line3 = plt.plot(x, x, 'g--', x, x**2, 'r:', x, x**3, 'b^')
line1.set_linewidth(3.0)
line1.set_dash_capstyle("round")
line3.set_alpha(0.2)
plt.show()


# ## Subplots
# A matplotlib figure may contain multiple subplots. These subplots are organized in a grid. To create a subplot, just call the `subplot` function, and specify the number of rows and columns in the figure, and the index of the subplot you want to draw on (starting from 1, then left to right, and top to bottom). Note that pyplot keeps track of the currently active subplot (which you can get a reference to by calling `plt.gca()`), so when you call the `plot` function, it draws on the *active* subplot.
# 

# In[105]:


x = np.linspace(-1.4, 1.4, 30)
plt.subplot(2, 2, 1)  # 2 rows, 2 columns, 1st subplot = top left
plt.plot(x, x)
plt.subplot(2, 2, 2)  # 2 rows, 2 columns, 2nd subplot = top right
plt.plot(x, x**2)
plt.subplot(2, 2, 3)  # 2 rows, 2 columns, 3rd subplot = bottow left
plt.plot(x, x**3)
plt.subplot(2, 2, 4)  # 2 rows, 2 columns, 4th subplot = bottom right
plt.plot(x, x**4)
plt.show()


# * Note that `subplot(223)` is a shorthand for `subplot(2, 2, 3)`.

# It is easy to create subplots that span across multiple grid cells like so:

# In[106]:


plt.subplot(2, 2, 1)  # 2 rows, 2 columns, 1st subplot = top left
plt.plot(x, x)
plt.subplot(2, 2, 2)  # 2 rows, 2 columns, 2nd subplot = top right
plt.plot(x, x**2)
plt.subplot(2, 1, 2)  # 2 rows, *1* column, 2nd subplot = bottom
plt.plot(x, x**3)
plt.show()


# If you need more complex subplot positionning, you can use `subplot2grid` instead of `subplot`. You specify the number of rows and columns in the grid, then your subplot's position in that grid (top-left = (0,0)), and optionally how many rows and/or columns it spans.  For example:

# In[107]:


plt.subplot2grid((3,3), (0, 0), rowspan=2, colspan=2)
plt.plot(x, x**2)
plt.subplot2grid((3,3), (0, 2))
plt.plot(x, x**3)
plt.subplot2grid((3,3), (1, 2), rowspan=2)
plt.plot(x, x**4)
plt.subplot2grid((3,3), (2, 0), colspan=2)
plt.plot(x, x**5)
plt.show()


# If you need even more flexibility in subplot positioning, check out the [GridSpec documentation](http://matplotlib.org/users/gridspec.html)

# When you are writing a program, *explicit is better than implicit*. Explicit code is usually easier to debug and maintain, and if you don't believe me just read the 2nd rule in the **Zen of Python**.

# You can write beautifully explicit code. Simply call the `subplots` function and use the figure object and the list of axes objects that are returned. No more magic! For example:

# In[108]:


x = np.linspace(-2, 2, 200)

fig1, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True)
fig1.set_size_inches(10,5)
line1, line2 = ax_top.plot(x, np.sin(3*x**2), "r-", x, np.cos(5*x**2), "b-")
line3, = ax_bottom.plot(x, np.sin(3*x), "r--")
ax_bottom.grid(True)


fig2, ax = plt.subplots(1, 1)
ax.plot(x, x**2)

plt.show()


# ## Drawing text
# You can call `text` to add text at any location in the graph. Just specify the horizontal and vertical coordinates and the text, and optionally some extra attributes.  Any text in matplotlib may contain TeX equation expressions, see [the documentation](http://matplotlib.org/users/mathtext.html) for more details.

# In[109]:


x = np.linspace(-1.5, 1.5, 30)
px = 0.8
py = px**2

plt.plot(x, x**2, "b-", px, py, "ro")

plt.text(0, 1.5, "Square function\n$y = x^2$", fontsize=20, color='blue', horizontalalignment="center")
plt.text(px - 0.08, py, "Beautiful point", ha="right", weight="heavy")
plt.text(px, py, "x = %0.2f\ny = %0.2f"%(px, py), rotation=-30, color='gray')

plt.show()


# * Note: `ha` is an alias for `horizontalalignment`
# 
# For more text properties, visit [the documentation](http://matplotlib.org/users/text_props.html#text-properties).
# 
# It is quite frequent to annotate elements of a graph, such as the beautiful point above. The `annotate` function makes this easy: just indicate the location of the point of interest, and the position of the text, plus optionally some extra attributes for the text and the arrow.

# In[110]:


plt.plot(x, x**2, px, py, "ro")
plt.annotate("Beautiful point", xy=(px, py), xytext=(px-1.3,py+0.5),
                           color="green", weight="heavy", fontsize=14,
                           arrowprops={"facecolor": "lightgreen"})
plt.show()


# You can also add a bounding box around your text by using the `bbox` attribute:

# In[111]:


plt.plot(x, x**2, px, py, "ro")

bbox_props = dict(boxstyle="rarrow,pad=0.3", ec="b", lw=2, fc="lightblue")
plt.text(px-0.2, py, "Beautiful point", bbox=bbox_props, ha="right")

bbox_props = dict(boxstyle="round4,pad=1,rounding_size=0.2", ec="black", fc="#EEEEFF", lw=5)
plt.text(0, 1.5, "Square function\n$y = x^2$", fontsize=20, color='black', ha="center", bbox=bbox_props)

plt.show()


# Just for fun, if you want an [xkcd](http://xkcd.com)-style plot, just draw within a `with plt.xkcd()` section:

# In[112]:


with plt.xkcd():
    plt.plot(x, x**2, px, py, "ro")

    bbox_props = dict(boxstyle="rarrow,pad=0.3", ec="b", lw=2, fc="lightblue")
    plt.text(px-0.2, py, "Beautiful point", bbox=bbox_props, ha="right")

    bbox_props = dict(boxstyle="round4,pad=1,rounding_size=0.2", ec="black", fc="#EEEEFF", lw=5)
    plt.text(0, 1.5, "Square function\n$y = x^2$", fontsize=20, color='black', ha="center", bbox=bbox_props)

    plt.show()


# ## Legends
# The simplest way to add a legend is to set a label on all lines, then just call the `legend` function.

# In[113]:


x = np.linspace(-1.4, 1.4, 50)
plt.plot(x, x**2, "r--", label="Square function")
plt.plot(x, x**3, "g-", label="Cube function")
plt.legend(loc="best")
plt.grid(True)
plt.show()


# ## Ticks and tickers
# The axes have little marks called "ticks".  To be precise, "ticks" are the *locations* of the marks (eg. (-1, 0, 1)), "tick lines" are the small lines drawn at those locations, "tick labels" are the labels drawn next to the tick lines, and "tickers" are objects that are capable of deciding where to place ticks. The default tickers typically do a pretty good job at placing ~5 to 8 ticks at a reasonable distance from one another.
# 
# But sometimes you need more control (eg. there are too many tick labels on the logit graph above). Fortunately, matplotlib gives you full control over ticks.  You can even activate minor ticks.
# 
# 

# In[114]:


x = np.linspace(-2, 2, 100)

plt.figure(1, figsize=(15,10))
plt.subplot(131)
plt.plot(x, x**3)
plt.grid(True)
plt.title("Default ticks")

ax = plt.subplot(132)
plt.plot(x, x**3)
ax.xaxis.set_ticks(np.arange(-2, 2, 1))
plt.grid(True)
plt.title("Manual ticks on the x-axis")

ax = plt.subplot(133)
plt.plot(x, x**3)
plt.minorticks_on()
ax.tick_params(axis='x', which='minor', bottom='off')
ax.xaxis.set_ticks([-2, 0, 1, 2])
ax.yaxis.set_ticks(np.arange(-5, 5, 1))
ax.yaxis.set_ticklabels(["min", -4, -3, -2, -1, 0, 1, 2, 3, "max"])
plt.title("Manual ticks and tick labels\n(plus minor ticks) on the y-axis")


plt.grid(True)

plt.show()


# ## 3D projection
# 
# Plotting 3D graphs is quite straightforward. You need to import `Axes3D`, which registers the `"3d"` projection. Then create a subplot setting the `projection` to `"3d"`. This returns an `Axes3DSubplot` object, which you can use to call `plot_surface`, giving x, y, and z coordinates, plus optional attributes.

# In[115]:


from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

figure = plt.figure(1, figsize = (12, 4))
subplot3d = plt.subplot(111, projection='3d')
surface = subplot3d.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='seismic', linewidth=0.1)
plt.show()


# Another way to display this same data is *via* a contour plot.

# In[116]:


plt.contourf(X, Y, Z, cmap='seismic')
plt.colorbar()
plt.show()


# ## Scatter plot

# To draw a scatter plot, simply provide the x and y coordinates of the points.

# In[117]:


x, y = np.random.rand(2, 100)
plt.scatter(x, y)
plt.show()


# You may also optionally provide the scale of each point.

# In[118]:


x, y, scale = np.random.rand(3, 100)
scale = 500 * scale ** 5
plt.scatter(x, y, s=scale)
plt.show()


# And as usual there are a number of other attributes you can set, such as the fill and edge colors and the alpha level.

# In[119]:


for color in ['red', 'green', 'blue']:
    n = 100
    x, y = np.random.rand(2, n)
    scale = 500.0 * np.random.rand(n) ** 5
    plt.scatter(x, y, s=scale, c=color, alpha=0.3, edgecolors='blue')

plt.grid(True)

plt.show()


# ## Histograms

# In[120]:


data = [1, 1.1, 1.8, 2, 2.1, 3.2, 3, 3, 3, 3]
plt.subplot(211)
plt.hist(data, bins = 10, rwidth=0.8)

plt.subplot(212)
plt.hist(data, bins = [1, 1.5, 2, 2.5, 3], rwidth=0.95)
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.show()


# In[121]:


data1 = np.random.randn(400)
data2 = np.random.randn(500) + 3
data3 = np.random.randn(450) + 6
data4a = np.random.randn(200) + 9
data4b = np.random.randn(100) + 10

plt.hist(data1, bins=5, color='g', alpha=0.75, label='bar hist') # default histtype='bar'
plt.hist(data2, color='b', alpha=0.65, histtype='stepfilled', label='stepfilled hist')
plt.hist(data3, color='r', histtype='step', label='step hist')
plt.hist((data4a, data4b), color=('r','m'), alpha=0.55, histtype='barstacked', label=('barstacked a', 'barstacked b'))

plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()


# # Example of beauty with matplotlib and NumPy

# In subsequent sections we'll provide a basic introduction to the nuts and bolts of the basic scientific python tools; but we'll first motivate it with a brief example that illustrates what you can do in a few lines with these tools.  For this, we will use the simple problem of approximating a definite integral with the trapezoid rule:
# 
# $$
# \int\limits_{a}^{b} f(x)\, dx \approx \frac{1}{2} \sum_{k=1}^{N} \left( x_{k} - x_{k-1} \right) \left( f(x_{k}) + f(x_{k-1}) \right).
# $$
# 
# Our task will be to compute this formula for a function such as:
# 
# $$
# f(x) = (x-3)(x-5)(x-7)+85
# $$
# 
# integrated between $a=1$ and $b=9$.
# 
# First, we define the function and sample it evenly between 0 and 10 at 200 points:

# In[225]:


f = lambda x: (x-3)*(x-5)*(x-7) + 85

x = np.linspace(0, 10, 200)
y = f(x)


# We select $a$ and $b$, our integration limits, and we take only a few points in that region to illustrate the error behavior of the trapezoid approximation:

# In[228]:


a, b = 1, 9
sampling = 10
xint = x[np.logical_and(x>=a, x<=b)][::sampling]
yint = y[np.logical_and(x>=a, x<=b)][::sampling]
# Fix end points of the interval
xint[0], xint[-1] = a, b
yint[0], yint[-1] = f(a), f(b)


# In[124]:


plt.plot([a, a], [0, f(a)], color='red')
plt.plot([b, b], [0, f(b)], color='red')
plt.plot(x, y, lw=2)
plt.axis([a-1, b+1, 0, 140])
plt.fill_between(xint, 0, yint, facecolor='gray', alpha=.4)
plt.text(0.5 * (a + b), 30,r"$\int_a^b f(x)\,dx$", horizontalalignment='center', fontsize=20);


# In[125]:


from scipy.integrate import quad, trapz

integral, error = quad(f, a, b)
trap_integral = trapz(yint, xint)
print("The integral is: %g +/- %.1e" % (integral, error))
print("The trapezoid approximation with", len(xint), "points is:", trap_integral)
print("The absolute error is:", abs(integral - trap_integral))


# This simple example showed us how, combining the numpy, scipy and matplotlib libraries we can provide an illustration of a standard method in elementary calculus with just a few lines of code.  We will now discuss with more detail the basic usage of these tools.
# 
# A note on visual styles: matplotlib has a rich system for controlling the visual style of all plot elements. [This page](https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html) is a gallery that illustrates how each style choice affects different plot types, which you can use to select the most appropriate to your needs.

# # Homework

# ## Problem 1

# Let $f(x) = 2\sin x - \cos{2x}$. Write a function `beauty_derivative_plot` that takes a point $x$ and plots the function $f(x)$ and its derivative $f'(x)$ in the range $x \in [-2\pi, 2\pi]$. The function should also plot the tangent line to the function at the point $x$. 
# 
# Example of output for $x = -3, -2, 1.7$:
# 
# ```python
# beauty_derivative_plot(-3)
# ```
# <img src="https://github.com/V-Kisielius/StatPrak24/blob/main/imgs/tangent_line_-3.0.png?raw=true" width="600"/>
# 
# ```python
# beauty_derivative_plot(-2)
# ```
# <img src="https://github.com/V-Kisielius/StatPrak24/blob/main/imgs/tangent_line_-2.0.png?raw=true" width="600"/>
# 
# ```python
# beauty_derivative_plot(1.7)
# ```
# <img src="https://github.com/V-Kisielius/StatPrak24/blob/main/imgs/tangent_line_1.7.png?raw=true" width="600"/>

# ### Let $f(x) = 2\sin x - \cos{2x}$. Write a function `beauty_derivative_plot` that takes a point $x$ and plots the function $f(x)$ and its derivative $f'(x)$ in the range $x \in [-2\pi, 2\pi]$. The function should also plot the tangent line to the function at the point $x$. 

# In[379]:


import numpy as np
import matplotlib.pyplot as plt

print('Hey!')
print('Enter the coordinates x of the tangent points in the range, separated by spaces:')

def f(x): 
    return 2*np.sin(x) - np.cos(2*x)

def slope(x): 
    return 2*np.cos(x) + 2*np.sin(2*x)

x = np.linspace(-2*np.pi, 2*np.pi, 100)

x_values = list(map(float, input().split()))

def line(x, x1, y1):
    return slope(x1)*(x - x1) + y1

def beauty_derivative_plot(x_value):
    print('beauty_derivative_plot (',x_value,')')
    x1 = x_value
    y1 = f(x1)
    plt.figure(figsize=(10, 6))
    plt.title('Function, Derivative, and Tangent Line at x = {:.2f}'.format(x1), fontsize=12)
    plt.plot(x, f(x), label="f(x) = sin(x) - 2cos(2x)",linewidth=2.5 )
    plt.plot(x, slope(x), label="f(x) = 2cos(x) + 2sin(2x)",linewidth=2.5)
    plt.scatter(x1, y1, color='C1')
    x1_str = "{:.2f}".format(x1)
    y1_str = "{:.2f}".format(y1)
    xrange = np.linspace(x1 - 1, x1 + 1, 10)
    plt.plot([0, x1], [y1, y1], 'k--', linewidth=1)
    plt.plot([x1, x1], [y1, 0], 'k--', linewidth=1)
    plt.plot(xrange, line(xrange, x1, y1), 'C1--', linewidth=2, label='tangent line in point {},{}'.format(x1_str, y1_str))
    plt.xlabel('Axe x')
    plt.ylabel('Axe y = f(x)')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

for x_value in x_values:
    beauty_derivative_plot(x_value)


# ## Problem 2

# In[284]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Добавляю библиотеки, чтобы считать прямо с ссылки


#  The data in [populations.txt](http://www.scipy-lectures.org/_downloads/populations.txt) describes the populations of hares, lynxes and carrots in northern Canada during 20 years. Get the data with 
# ```python
# np.loadtxt('http://www.scipy-lectures.org/_downloads/populations.txt')
# ```
# and do the following exercises **without for-loops**.

# ### Plot the population size of each species for each year on a single graph. Add legend and axis labels to the plot.

# In[372]:


df = pd.read_csv('http://scipy-lectures.org/_downloads/populations.txt', sep='\t')
df_long = df.melt(id_vars=['# year'])
plt.figure(figsize=(10, 6), dpi=150)

sns.lineplot(df_long, x='# year', y='value', hue='variable')
plt.title('Population size by year')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid(True)
plt.legend(title='Type')
plt.show()


# ### Find which year each species had the largest population. Mark these years in the plot of populations.

# In[373]:


df_long = df.melt(id_vars=['# year'], var_name='Sort', value_name='Numbers')

max_populations_idx = df_long.groupby('Sort')['Numbers'].idxmax()
max_years = df_long.loc[max_populations_idx]

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_long, x='# year', y='Numbers', hue='Sort')

for idx, row in max_years.iterrows():
    plt.text(row['# year'], row['Numbers'], str(int(row['# year'])), ha='center', va='bottom')

plt.title('Population size by year')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend(title = 'Type')
plt.grid(True)
plt.show()


# ### Assuming total population of hares and lynxes is 100%, plot stacked bar graph showing ratio of each specis for each year in the period.

# In[374]:


df['total'] = df.iloc[:, 1:].sum(axis=1)

plt.figure(figsize=(10, 6))
plt.bar(df['# year'], df['hare'], color='C1', alpha=0.5, label='Hare', align='center')

plt.bar(df['# year'], df['lynx'], bottom=df['hare'], alpha=0.5, label='Lynx', align='center')

plt.title('Persantage of lynx and hate ')
plt.xlabel('Year')
plt.ylabel('Amount of population')
plt.legend()
plt.grid(True)
plt.show()


# ### Find the mean and std of the populations of each species. Plot the historgam of population for each species and show mean values with vertical line. Show each histogram in a separate plot. Arrange 3 plots in a row.

# In[375]:


df = pd.read_csv('http://scipy-lectures.org/_downloads/populations.txt', sep='\t')

mean_hare = df['hare'].mean()
std_hare = df['hare'].std()

mean_lynx = df['lynx'].mean()
std_lynx = df['lynx'].std()

mean_carrot = df['carrot'].mean()
std_carrot = df['carrot'].std()


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(df['hare'], bins=10, color='Purple', alpha=0.5, edgecolor='black')
plt.axvline(mean_hare, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_hare:.2f}')
plt.title('Population of hares')
plt.xlabel('Population')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(df['lynx'], bins=10, color='Navy', alpha=0.5, edgecolor='black')
plt.axvline(mean_lynx, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_lynx:.2f}')
plt.title('Population of lynx')
plt.xlabel('Population')
plt.ylabel('Frequency')
plt.legend()
          
plt.subplot(1, 3, 3)
plt.hist(df['carrot'], bins=10, color='Teal', alpha=0.5, edgecolor='black')
plt.axvline(mean_carrot, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_carrot:.2f}')
plt.title('Amount of carrot')
plt.xlabel('Population')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()



# ### Find which species (hares or lynxes) has the largest population for each year. Print the result as [H, H, L, H, ...]. Plot a pie chart showing the ratio of "H" and "L" values.

# In[382]:


max_species = df[['hare', 'lynx']].idxmax(axis=1)
max_species = max_species.apply(lambda x: 'H' if x == 'hare' else 'L')

print(max_species.tolist())

plt.figure(figsize=(6,6))
max_species.value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['Coral', 'Royalblue'], labels=['Hare', 'Lynx'])
plt.title('Percentage of each graph')
plt.ylabel('')
plt.legend()
plt.show()


# ### Plot the change in the hare population and in the lynx population for each year. Find the correlation coefficient between the both time series.

# In[321]:


df_long = df.melt(id_vars=['# year'])

df_long = df_long[df_long['variable'].isin(['lynx', 'hare'])]
plt.figure(figsize=(10, 6), dpi=150)

sns.lineplot(df_long, x='# year', y='value', hue='variable')
plt.title('Population size by year')
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend(title='Type')
plt.show()

df['hare_change'] = df['hare'].diff()
df['lynx_change'] = df['lynx'].diff()

correlation = df['hare_change'].corr(df['lynx_change'])

print("Сorrelation coefficient is ~ ", abs(correlation))


# ### Using a scatter plot, show the population of hares vs lynxes by year (each point corresponds to a particular year, and the point coordinates correspond to the population of the two species in that year).

# In[381]:


plt.figure(figsize=(10, 6))
plt.scatter(df['# year'], df['hare'], label='Hare', color='Coral')
plt.scatter(df['# year'], df['lynx'], label='Lynx', color='royalblue')

plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population of Hare and Lynxe by Year')
plt.legend()

plt.grid(True)
plt.show()


# ### Assume the population of hares in 1920 is unknown. Suggest a way to estimate this value. Compare an estimated value with the true value and print a ratio of the error to the true value.

# Predict based on the difference of previous two results

# In[343]:


diff_hare_1919_1918 = df.loc[df['# year'] == 1919, 'hare'].values[0] - df.loc[df['# year'] == 1918, 'hare'].values[0]

estimated_hare_1920 = df.loc[df['# year'] == 1919, 'hare'].values[0] + diff_hare_1919_1918

true_hare_1920 = df.loc[df['# year'] == 1920, 'hare'].values[0]


print("Estimated group of hares in 1920:", estimated_hare_1920)
print("Real number of hares in 1920:", true_hare_1920)
print("Ratio of error to true value:", abs(estimated_hare_1920 - true_hare_1920) / true_hare_1920)
print("Difference between error and true value", abs(estimated_hare_1920 - true_hare_1920))


# In[ ]:




