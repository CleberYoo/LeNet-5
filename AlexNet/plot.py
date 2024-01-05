import matplotlib.pyplot as plt
import numpy as np


def numpy_image_show(image):
    image = image.mean(dim=0)
    image = image / 2 + 0.5

    numpy_image = image.numpy()

    plt.imshow(X=numpy_image)



def image_show(image):
    image = image.mean(dim=0)
    image = image / 2 + 0.5

    plt.imshow(X=image)