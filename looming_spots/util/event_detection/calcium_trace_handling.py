import numpy as np


def load_ratio(green_path):
    header, greens, reds = load_green_red(green_path)
    cells_ratios = greens[:, :, 0] / reds[:, :, 0]
    return cells_ratios


def load_green_red(green_path):
    red_path = green_path[:-5] + "2.csv"
    header, greens = load_fluo_profiles(green_path)
    _, reds = load_fluo_profiles(red_path)
    return header, greens, reds


def load_fluo_profiles(profiles_path):
    """
    Assumes structure:

    .. csv-table:
        cell0        1.5349583   2.324254    3.45353243     1.4234455     ...    3.23423453
        cell1        1.2345354   2.345435    2.34534546     3.3425365     ...    4.31232453
        cell2        2.2345365   4.345346    3.89324235     2.3453645     ...    3.23423453
        neuropile0   1.5349583   2.324254    3.45353243     1.4234455     ...    1.23423453
        neuropile1   1.2345354   2.345435    2.34534546     3.3425365     ...    3.23443533
        neuropile2   2.2345365   4.345346    3.89324235     2.3453645     ...    3.23478978

    :returns: a matrix with shape(image0:nImages, cell0:n, mType cell:neuropile)
    """

    with open(profiles_path, "r") as in_file:
        data = in_file.readlines()
    data = [d.strip() for d in data]
    data = [d.split("\t") for d in data]
    header = [d[0] for d in data]

    data = [d[1:] for d in data]
    try:
        data = np.array(data, dtype=np.float64)
    except ValueError:
        print("Error with {}".format(profiles_path))
        raise
    data = data.conj().transpose()  # flip (matlab a' equivalent)

    cells_idx = [i for i in range(len(header)) if header[i].startswith("cell")]
    neuropiles_idx = [
        i for i in range(len(header)) if header[i].startswith("neuropile")
    ]

    if cells_idx:
        cells = data[:, cells_idx[0] : cells_idx[-1] + 1]
    if neuropiles_idx:
        neuropiles = data[:, neuropiles_idx[0] : neuropiles_idx[-1] + 1]

    if cells_idx and neuropiles_idx:
        if not cells.size == neuropiles.size:
            raise ValueError(
                "Cells and neuropiles must have same length in {}".format(
                    profiles_path
                )
            )
        data = np.dstack(
            (cells, neuropiles)
        )  # concatanate cells and neuropile in z
    else:
        if cells_idx:
            data = cells
        elif neuropiles_idx:
            data = neuropiles
        else:
            raise ValueError("Missing data in {}".format(profiles_path))
    return header, data
