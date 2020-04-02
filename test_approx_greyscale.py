from piecewise_model import PWAModel

import multiprocessing
import cv2

import matplotlib.pyplot as plt
import numpy as np

from functools import partial

def load_image():
    image = cv2.imread('/home/cannon/Documents/piecewise_rls_approximation/lovett.jpg')
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(np.max(grayscale))

    return grayscale

def sample_pixel(image):
    x = np.random.randint(0, high=image.shape[0])
    y = np.random.randint(0, high=image.shape[1])

    loc = np.array([float(y)/float(image.shape[1]), 1.0 - float(x)/float(image.shape[0])]).reshape((2, 1))
    pixel = np.array([float(image[x, y])/float(255)]).reshape((1, 1))

    return (loc, pixel)

def float_to_grayscale(flt):
    return int(round(flt[0][0] * 255))

def float_loc_to_pixel_coords(loc, image):
    x = int(round(loc[0][0] * (image.shape[0] - 1)))
    y = int(round(loc[1][0] * (image.shape[1] - 1)))

    return (x, y)

def compute_image_row(model, X, Y, nx, ny, i):
    row = np.zeros((ny,))
    for j in range(ny):
        input_vec = np.array([X[i,j], Y[i,j]]).reshape((2,1))
        row[j] = model.predict(input_vec)

    return row

def plot_model(model, ax, title, pool):

    x = np.linspace(0.0, 1.0, 768)
    y = np.linspace(0.0, 1.0, 768)

    nx = len(x)
    ny = len(y)

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

def run():
    image = load_image()
    pwa_model = PWAModel(1)

    p = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    iteration = 0
    while iteration < 1e8:
        print("On iteration {}".format(iteration))
        iteration += 1

        loc, pixel = sample_pixel(image)
        print("{}: {}".format(loc, pixel))
        print("{}; {}".format(float_loc_to_pixel_coords(loc, image), float_to_grayscale(pixel)))

        pwa_model.process_datum(loc, pixel)

        if iteration % 1000 == 0:
            plot_model(pwa_model, plt.gca(), "Fit PWA Model After {} Samples".format(iteration), p)
            #plt.show()
            plt.savefig('plots/approx_{}.png'.format(iteration), dpi=100)
            plt.gca().clear()

        if iteration % 100 == 0 and iteration > 0:
            pwa_model.remove_random_ref()

if __name__ == "__main__":
    run()
