# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as ssp


def MaxNormalise(corpus_mtrx, vocab_len):

    # Getting the Maximum frequency for every document.
    max_val = np.max(corpus_mtrx.todense())

    if max_val == 0.0:

        # NOTE: Preventing division-by-zero For Documents with zero terms. This case occurs...
        # when a sub-Vocabulary is used for the experiment.
        max_val = 1.0

        # NOTE: PATCH for preventing All-Zero-Values vectors stopping the experiments.
        corpus_mtrx[:] = 1e-15

    # Normalizing based on the matrix/array type.
    if ssp.issparse(corpus_mtrx):
        corpus_mtrx = ssp.csr_matrix(corpus_mtrx.todense() / max_val)
    else:
        corpus_mtrx = corpus_mtrx / max_val

    return corpus_mtrx


def SubSamplingNorm(corpus_mtrx, vocab_len, sb_t=0.0001):

    # Getting the Maximum frequency for every document.
    max_val = np.max(corpus_mtrx.todense())

    # NOTE: PATCH for preventing All-Zero-Values vectors stopping the experiments.
    corpus_mtrx[np.where(corpus_mtrx == 0.0)] = 1e-15

    # Applying SubSampling with pre Max-Normalization.
    if ssp.issparse(corpus_mtrx):
        corpus_mtrx = ssp.csr_matrix(1 - np.sqrt(sb_t / (corpus_mtrx.todense() / max_val)))
    else:
        corpus_mtrx = 1 - np.sqrt(sb_t / (corpus_mtrx / max_val))

    return corpus_mtrx
