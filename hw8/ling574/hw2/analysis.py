import random
import time

import sklearn
import numpy as np
import pandas as pd
import plotnine as pn

from sklearn.neighbors import NearestNeighbors

import util
from vocabulary import Vocabulary


if __name__ == "__main__":

    args = util.get_args()

    vectors = util.read_vectors(args.save_vectors)

    vector_matrix = np.stack([vectors[word] for word in vectors])
    vector_words = list(vectors.keys())

    vocab = Vocabulary.from_text_files([args.training_data], min_freq=args.min_freq)

    visualize_words = [
        "great",
        "cool",
        "brilliant",
        "amazing",
        "sweet",
        "enjoyable",
        "boring",
        "bad",
        "dumb",
        "annoying",
        "female",
        "male",
        "queen",
        "king",
        "man",
        "woman",
        "director",
    ]

    visualize_idxs = [vector_words.index(token) for token in visualize_words]

    pca = sklearn.decomposition.PCA(n_components=2).fit_transform(vector_matrix)

    vector_dict = {
        "word": visualize_words,
        "x": pca[visualize_idxs][:, 0],
        "y": pca[visualize_idxs][:, 1],
    }

    projected_frame = pd.DataFrame(vector_dict)

    plot = (
        pn.ggplot(projected_frame, pn.aes(x="x", y="y", label="word")) + pn.geom_text()
    )
    plot.save(args.save_plot)