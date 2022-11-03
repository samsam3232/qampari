import random

random.seed(42)

def randomly_merge(positive_ctxs, negative_ctxs):

    """
    Randomly merges the positive and the negative contexts during train time.
    :param positive_ctxs: list of positive contexts
    :param negative_ctxs: list of negative contexts
    :return: the randomly mixed contexts
    """

    final_ctxs = list()
    while positive_ctxs and negative_ctxs:
        # choosing randomly from which list to pick the first element
        if bool(random.getrandbits(1)):
            final_ctxs.append(positive_ctxs.pop(0))
        else:
            final_ctxs.append(negative_ctxs.pop(0))
    final_ctxs += positive_ctxs + negative_ctxs

    return final_ctxs