# -*- coding: utf-8

import os
import joblib
import xgboost as xgb
from mol_property.pka.data_utils import DataUtils


class PkaPredictor(object):
    def __init__(
        self,
        model_dir=os.path.join(os.path.dirname(__file__), "model"),
        feature_type="morgan+macc",
        allow_multilabel=True,
    ):
        self.allow_multilabel = allow_multilabel
        self._load_model(
            clf_multi_modelpath=os.path.join(model_dir, "pka_classification.pkl"),
            clf_modelpath=os.path.join(model_dir, "pka_classification.txt"),
            acidic_modelpath=os.path.join(model_dir, "pka_acidic_regression.txt"),
            basic_modelpath=os.path.join(model_dir, "pka_basic_regression.txt"),
        )
        self.feature_type = feature_type

    def _load_model(
        self, clf_modelpath, clf_multi_modelpath, acidic_modelpath, basic_modelpath
    ):
        self.acidic_reg = xgb.XGBRegressor()
        self.acidic_reg.load_model(acidic_modelpath)
        self.basic_reg = xgb.XGBRegressor()
        self.basic_reg.load_model(basic_modelpath)
        if self.allow_multilabel:
            self.clf = joblib.load(clf_multi_modelpath)
        else:
            self.clf = xgb.XGBClassifier()
            self.clf.load_model(clf_modelpath)

    def predict(self, mols):
        mols_features = DataUtils.get_molecular_features(mols, self.feature_type)
        clf_labels = self.clf.predict(mols_features)
        acidic_scores = self.acidic_reg.predict(mols_features)
        basic_scores = self.basic_reg.predict(mols_features)
        rets = []
        for idx, clf_label in enumerate(clf_labels):
            ret = {}
            if self.allow_multilabel:
                if clf_label[0] == 1:
                    ret["acidic"] = acidic_scores[idx]
                if clf_label[1] == 1:
                    ret["basic"] = basic_scores[idx]
            else:
                if clf_label == 0:
                    ret["acidic"] = acidic_scores[idx]
                elif clf_label == 1:
                    ret["basic"] = basic_scores[idx]
            rets.append(ret)
        return rets
