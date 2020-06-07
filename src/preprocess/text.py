import gzip
import os
import re
import random

from dataset import SmoteDataset
from preprocess.uci import PREPROC_REG


class Preprocessor:
    def __init__(self, mode):
        assert mode in set(self._FILE_PREPROC_MAP.keys()).union(
            set(PREPROC_REG.keys())), mode
        self.mode = mode

    def preprocess_file(self, file_path, out_path=None):
        with open(file_path) as in_f:
            text = in_f.read()
        return PREPROC_REG[self.mode](text)

    def preprocess_text(self, text):
        return PREPROC_REG[self.mode](text)
