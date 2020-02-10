import pathlib


def get_package_dir():
    """
    Get the directory of this script
    (which also happens to be the package directory)
    Might be just a simple function, but saves a lot of headaches
    """
    return pathlib.Path(__file__).parent.absolute()

