import matplotlib
import matplotlib.pyplot as plt
from util.leica import location, Dimensions

matplotlib.use('tkagg')


def make_plot(mlist: list[int], dims: Dimensions, directory: str) -> list[list[float]]:
    imx = 1314.09 / 1000
    imy = 875.89 / 1000  # mm
    parr = []
    plt.figure(figsize=(18, 18))
    print(mlist)

    for m in mlist:
        x, y, max_y, max_x = location(m, dims)
        print(x, y, max_y, max_x)
        plt.scatter(x * imx, y * imy)
        plt.text(x * imx, y * imy + .03, m, fontsize=9)
        parr.append([m, round(x * imx, 1), round(y * imy, 1)])

    bound_x = [0, max_x * imx, max_x * imx, 0, 0]
    bound_y = [0, 0, max_y * imy, max_y * imy, 0]

    plt.plot(bound_x, bound_y)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.savefig(directory + "coordmap.jpg")
    plt.close()

    return parr
