import click
import pandas as pd
import mppcca
import utils
import numpy as np


@click.command()
@click.option("-i", "--in_filename", help="input csv filename")
@click.option("-k", help="number of clusters", type=int)
@click.option("-d", "--delay", help="delay time", type=int)
@click.option("-e", "--embedding", help="embedding time", type=int)
@click.option("-o", "--out_filename", help="outputfilename")
def mppcca_from_csv(in_filename, k, delay, embedding, out_filename):

    print("input filename: \t", in_filename)
    print("output filename:\t", out_filename)
    print("delay:\t\t\t", delay)
    print("embedding:\t\t", embedding)
    print("K:\t\t\t", k)

    for option in [in_filename, k, delay, embedding, out_filename]:
        if option == None:
            print("set all options!")
            exit(-1)

    data = pd.read_csv(in_filename)
    x = np.array(data["x"]).reshape(-1, 1)
    y = np.array(data["y"]).reshape(-1, 1)

    xt_1 = utils.embed(x, delay, embedding)
    yt_1 = utils.embed(y, delay, embedding)
    yt = y[delay + embedding:]
    predicted_params, predicted_labels = mppcca.mppcca(yt, xt_1, yt_1, k)

    data["labels"] = np.r_[[None] * (delay + embedding), predicted_labels]

    data.to_csv(out_filename, index=False)


if __name__ == '__main__':
    mppcca_from_csv()
