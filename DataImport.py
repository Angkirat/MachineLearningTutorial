import pandas as pd


def csvPandas(location, header=True, separator=","):
    dataOutput = pd.read_csv(location, header=header, sep=separator)
    return dataOutput
    pass

