import numpy as np
import argparse


def dim_get(input_dir: str) -> tuple[int, int]:
    """
    Gets the scan dimensions from a microscope file.
    :param input_dir: The directory containing the microscope file.
    :return: The dimensions of the scan, as a tuple of [x, y].
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
            rawdata3 = rawdata2[2].partition('" FieldY="')
            xd = int(rawdata3[0])
            rawdata4 = rawdata3[2].partition('" PosX="')
            yd = int(rawdata4[0])
            rawdata6 = rawdata4[2].partition('" PosY="')
            posx = float(rawdata6[0])
            rawdata8 = rawdata6[2].partition('" PosZ="')
            posy = float(rawdata8[0])
            rawdata10 = rawdata8[2].partition('" />')
            rawdata = rawdata10[2]
            posarr.append([xd, posx, yd, posy])
            size = len(rawdata)
            # print(size)
        posarr = np.array(posarr)
        print('Position dict made')
        return posarr


if __name__ == '__main__':
    from config import load_config

    # TODO: new description or abstract
    parser = argparse.ArgumentParser(
        description="Find graphene flakes on SiO2. Currently configured only for exfoliator dataset"
    )
    parser.add_argument(
        "--q",
        required=True,
        type=str,
        help="Directory containing images to process. Optional unless running in headless mode"
    )
    args = parser.parse_args()

    for input_dir, _ in load_config(args.q):
        print(f"Found dimensions for {input_dir}: {dim_get(input_dir)}")
