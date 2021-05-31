from matplotlib import pyplot as plt


def print_sinusoid(p):
    plt.plot(p)
    plt.title("Centroid coordinates over axe y")
    plt.savefig("test.png")