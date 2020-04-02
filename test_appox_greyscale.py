from piecewise_model import PWAModel

import cv2

import matplotlib.pyplot as plt
import numpy as np

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

def run():
    image = load_image()
    pwa_model = PWAModel()

    iteration = 0
    while iteration < 1e8:
        print("On iteration {}".format(iteration))
        iteration += 1

        loc, pixel = sample_pixel(image)
        print("{}: {}".format(loc, pixel))
        print("{}; {}".format(float_loc_to_pixel_coords(loc, image), float_to_grayscale(pixel)))

        pwa_model.process_datum(loc, pixel)

        if iteration % 1000 == 0:
            plt.figure(figsize=(8, 6))
            pwa_model.plot_model(plt.gca(), "Fit PWA Model After {} Samples".format(iteration))
            #plt.show()
            plt.savefig('plots/approx_{}.png'.format(iteration), dpi=100)
            plt.close()

        if iteration % 100 == 0 and iteration > 0:
            pwa_model.remove_random_ref()

if __name__ == "__main__":
    run()
