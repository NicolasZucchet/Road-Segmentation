import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def weights_comparison(models:list, step=50):
    ''' Calculates cosine-similarity between weights of different models
        and plots a similarity-score matrix 

        models: list of torch models
        step:   step size between ticks in labels '''

    weights = []
    for m in models:
        weights.append(torch.nn.utils.parameters_to_vector(m.parameters()).cpu().detach().numpy())
    weights = np.array(weights)

    mat = cosine_similarity(weights)[::-1]

    fig, ax = plt.subplots(figsize=(15,15))

    im = ax.imshow(mat)

    ax.set_xticks(np.arange(start=0, stop=len(models), step=1))
    ax.set_yticks(np.arange(start=0, stop=len(models), step=1))
    ax.set_xticklabels(np.arange(start=0, stop=len(models)*step, step=step), fontsize=40)
    ax.set_yticklabels(np.arange(start=len(models)*(step-1), stop=-step, step=-step), fontsize=40)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    for i in range(len(models)):
        for j in range(len(models)):
            text = ax.text(j, i, round(mat[i, j], 3),
                        ha="center", va="center", color="w", fontsize=40)

    fig.tight_layout()
    plt.show()