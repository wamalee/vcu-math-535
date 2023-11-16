import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from PIL import Image, ImageDraw
import glob

# Create an image of the Lorenz attractor.
# The maths behind this code is described in the scipython blog article
# at https://scipython.com/blog/the-lorenz-attractor/
# Christian Hill, January 2016.
# Updated, January 2021 to use scipy.integrate.solve_ivp.

WIDTH, HEIGHT, DPI = 1000, 750, 100

# Lorenz paramters and initial conditions.
sigma, beta, rho = 10, 2.667, 126.52
u0, v0, w0 = 0, 1, 1.05

parmMax = 2
master = np.arange(-0, 50, 1)

# Maximum time point and total number of time points.
tmax, n = 100, 10000

def lorenz(t, X, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

count = 0
for rho in master:

    # Integrate the Lorenz equations.
    soln = solve_ivp(lorenz, (0, tmax), (u0, v0, w0), args=(sigma, beta, rho),
                    dense_output=True)
    # Interpolate solution onto the time grid, t.
    t = np.linspace(0, tmax, n)
    x, y, z = soln.sol(t)

    # Plot the Lorenz attractor using a Matplotlib 3D projection.
    fig = plt.figure(facecolor='k', figsize=(WIDTH/DPI, HEIGHT/DPI))
    #ax = fig.gca(projection='3d')
    #ax = fig.gca()
    ax = fig.add_subplot(projection='3d')
    ax.set_facecolor('k')
    ax.grid(True)
    plt.title('Lorenz Attractor rho='+str(rho), color = 'white')
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Make the line multi-coloured by plotting it in segments of length s which
    # change in colour across the whole time series.
    s = 10
    cmap = plt.cm.winter
    for i in range(0,n-s,s):
        ax.plot(x[i:i+s+1], y[i:i+s+1], z[i:i+s+1], color=cmap(i/n), alpha=0.4)

    # Remove all the axis clutter, leaving just the curve.
    # ax.set_axis_off()

    print(rho)
    #plt.show()
    plt.savefig('lorenz itr'+str(f'{count:03}')+'.png', dpi=DPI)
    count = count + 1
    #plt.show()
    plt.clf()
    plt.close()


fp_in = "Lorenz itr*.png"
fp_out = "mygif.gif"

img, *imgs = [Image.open(f) for f in glob.glob(fp_in)]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=500, loop=0)