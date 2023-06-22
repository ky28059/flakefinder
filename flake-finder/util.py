import numpy as np


def bg_to_flake_color(rgbarr):
    """
    Returns the flake color based on an input background color. Values determined empirically.
    :param rgbarr: The RGB array representing the color of the background.
    :return: The RGB array representing the color of the flake.
    """
    red, green, blue = rgbarr
    rval = int(round(0.8643 * red - 2.55, 0))
    gval = int(round(0.8601 * green + 9.6765, 0))
    bval = blue + 4
    # print('coloring')
    return np.array([rval, gval, bval])


def dim_get(input_dir: str) -> tuple[int, int]:
    """
    Gets the scan dimensions from a microscope file.
    :param input_dir: The directory containing the microscope file.
    :return: The dimensions of the scan, as [x, y].
    """
    try:
        with open(input_dir + "/leicametadata/TileScan_001.xlif", 'r') as file:
            rawdata = file.read()
        rawdata2 = rawdata.partition('</Attachment>')
        rawdata = rawdata2[0]
        size = len(rawdata)
        xarr = []
        yarr = []
        while size > 10:
            rawdata2 = rawdata.partition('FieldX="')
            rawdata = rawdata2[2]
            rawdata3 = rawdata.partition('" FieldY="')
            xd = int(rawdata3[0])
            rawdata = rawdata3[2]
            rawdata4 = rawdata.partition('" PosX="')
            yd = int(rawdata4[0])
            rawdata = rawdata4[2]
            rawdata5 = rawdata.partition('" />')
            rawdata = rawdata5[2]
            xarr.append(xd)
            yarr.append(yd)
            size = len(rawdata)
            # print(size)
        xarr = np.array(xarr)
        yarr = np.array(yarr)
        print('Your scan is ' + str(np.max(xarr) + 1) + ' by ' + str(np.max(yarr) + 1))
        return np.max(xarr) + 1, np.max(yarr) + 1
    except:
        return 1, 1


def pos_get(input_dir: str):  # finds image location from microscope file
    filename = input_dir + "/leicametadata/TileScan_001.xlif"
    print(filename)
    posarr = []
    with open(filename, 'r') as file:
        rawdata = file.read()
        rawdata2 = rawdata.partition('</Attachment>')
        rawdata = rawdata2[0]
        size = len(rawdata)
        while size > 10:
            rawdata2 = rawdata.partition('FieldX="')
            rawdata = rawdata2[2]
            rawdata3 = rawdata.partition('" FieldY="')
            xd = int(rawdata3[0])
            rawdata = rawdata3[2]
            rawdata4 = rawdata.partition('" PosX="')
            yd = int(rawdata4[0])
            rawdata5 = rawdata4[2]
            rawdata6 = rawdata5.partition('" PosY="')
            posx = float(rawdata6[0])
            rawdata7 = rawdata6[2]
            rawdata8 = rawdata7.partition('" PosZ="')
            posy = float(rawdata8[0])
            rawdata9 = rawdata8[2]
            rawdata10 = rawdata9.partition('" />')
            rawdata = rawdata10[2]
            posarr.append([xd, posx, yd, posy])
            size = len(rawdata)
            # print(size)
        posarr = np.array(posarr)
        print('Position dict made')
        return posarr
