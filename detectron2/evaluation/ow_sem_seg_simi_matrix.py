import os
import numpy as np
import pandas as pd

class OWSemSegSimiMatrix:
    def __init__(self, path=None):
        assert os.path.exists(path), 'similarity matrix {} is not exists.'.format(path)
        df_sim = pd.read_csv(path, index_col=0)
        self.matrix = df_sim.values
        self.labels = list(df_sim.columns)

    def findSimiElement(self, dt_label, gt_label):
        assert str(dt_label) in self.labels, "Category id not belong to this dataset."
        assert str(gt_label) in self.labels, "Category id not belong to this dataset."

        if dt_label == gt_label:
            return 1
        else:
            simi = self.matrix[dt_label, gt_label]
            return simi

    def findSimiMatrix(self):
        return self.matrix

    def findLabelList(self):
        return self.labels
    
class SIMIaccess:
    def __init__(self, path=None):
        assert os.path.exists(path), 'similarity matrix {} is not exists.'.format(path)
        df_sim = pd.read_csv(path, index_col=0)
        self.matrix = df_sim.values
        self.labels = list(df_sim.columns)
        self.label_to_index = dict()
        for i, label in enumerate(self.labels):
            self.label_to_index.update({label: i})

    def findSimiElement(self, dt_label, gt_label):
        dt_label = str(dt_label)
        gt_label = str(gt_label)
        assert dt_label in self.labels, "Category id not belong to this dataset."
        assert gt_label in self.labels, "Category id not belong to this dataset."

        if dt_label == gt_label:
            return 1
        else:
            dt_index = self.label_to_index.get(dt_label)
            gt_index = self.label_to_index.get(gt_label)
            simi = self.matrix[dt_index, gt_index]
            return simi

    def findSimiMatrix(self):
        return self.matrix