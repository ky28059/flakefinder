import matplotlib
import matplotlib.pyplot as plt
from leica import Dimensions

matplotlib.use('tkagg')


def location(m: int, dims: Dimensions) -> tuple[float, int, int, int]:  # TODO: supposed to be a float?
    width, height = dims
    row = m % height
    column = (m - row) / height
    print(f"Location for {m}: {column}, {row}")
    return column, row, height - 1, width - 1


def make_plot(mlist, dims: Dimensions, directory: str) -> list[list[float]]:
    imx = 1314.09 / 1000
    imy = 875.89 / 1000  # mm
    parr = []
    plt.figure(figsize=(18, 18))
    print(mlist)

    for m in mlist:
        x, y, maxy, maxx = location(m, dims)
        print(x, y, maxy, maxx)
        plt.scatter(x * imx, y * imy)
        plt.text(x * imx, y * imy + .03, m, fontsize=9)
        parr.append([m, round(x * imx, 1), round(y * imy, 1)])

    boundx = [0, maxx * imx, maxx * imx, 0, 0]
    boundy = [0, 0, maxy * imy, maxy * imy, 0]

    plt.plot(boundx, boundy)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.savefig(directory + "coordmap.jpg")
    plt.close()

    return parr
