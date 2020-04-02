from piecewise_model import PWAModel

import multiprocessing
import cv2

import matplotlib.pyplot as plt
import numpy as np

from functools import partial

import sys

def load_image(filename):
    image = cv2.imread(filename)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(rgb[0, 0, :])
    print(rgb.shape)

    return rgb, image.shape[0], image.shape[1]

def sample_pixel(image):
    x = np.random.randint(0, high=image.shape[0])
    y = np.random.randint(0, high=image.shape[1])

    loc = np.array([1.0 - float(x)/float(image.shape[0]), float(y)/float(image.shape[1])]).reshape((2, 1))
    pixel = np.array([image[x, y,:]/float(255)]).reshape((3, 1))

    return (loc, pixel)

def float_to_rgb(flt):
    return np.array([int(round(flt[0][0] * 255)), int(round(flt[1][0] * 255)), int(round(flt[2][0] * 255))])

def float_loc_to_pixel_coords(loc, image):
    x = int(round(loc[0][0] * (image.shape[0] - 1)))
    y = int(round(loc[1][0] * (image.shape[1] - 1)))

    return (x, y)

def compute_image_row(model, X, Y, nx, ny, i):
    row = np.zeros((ny,3))
    for j in range(ny):
        input_vec = np.array([X[j, i], Y[j, i]]).reshape((2,1))
        row[j,:] = model.predict(input_vec).reshape((3,))

    return row

def plot_model(model, ax, nx, ny, title, pool):

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)

    X, Y = np.meshgrid(x, y)
    f = partial(compute_image_row, model, X, Y, nx, ny)
    rows = pool.map(f, range(nx))
    Z = np.stack(rows[::-1])

    # Old version
    #Z = np.zeros_like(X)
    #for i in range(nx):
    #    for j in range(ny):
    #        input_vec = np.array([X[i,j], Y[i,j]]).reshape((2,1))
    #        Z[nx - i - 1, j] = model.predict(input_vec)


    #plt.xlim(0., 1.)
    #plt.ylim(0., 1.)
    #plt.contourf(x, y, Z, levels=255, cmap='gray')
    #plt.clim(0., 1.)
    ax.imshow(Z, cmap='gray', vmin=0.0, vmax=1.0)
    ax.set_title(title)

    #ref_x = []
    #ref_y = []
    #for ref_point in model.ref_points:
    #    ref_x.append(ref_point[0])
    #    ref_y.append(ref_point[1])

    #ax.scatter(ref_x, ref_y, c='k')

def run(image_filename):
    image, nx, ny = load_image(image_filename)
    pwa_model = PWAModel(3)

    p = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    iteration = 0
    while iteration < 1e8:
        print("On iteration {}".format(iteration))
        iteration += 1

        loc, pixel = sample_pixel(image)
        print("{}: {}".format(loc, pixel))
        print("{}; {}".format(float_loc_to_pixel_coords(loc, image), float_to_rgb(pixel)))

        pwa_model.process_datum(loc, pixel)

        if iteration % 1000 == 0:
            plot_model(pwa_model, plt.gca(), nx, ny, "Fit PWA Model After {} Samples".format(iteration), p)
            #plt.show()
            plt.savefig('plots/approx_{}.png'.format(int(iteration/1000)), dpi=100)
            plt.gca().clear()

        if iteration % 100 == 0 and iteration > 0:
            pwa_model.remove_random_ref()

if __name__ == "__main__":
    assert(len(sys.argv) == 2)
    run(sys.argv[1])
    run()
