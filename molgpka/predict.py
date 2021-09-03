#!/usr/bin/env python
# coding: utf-8

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem.MolStandardize import rdMolStandardize

import requests
import random
import os
import os.path as osp
import numpy as np
import pandas as pd

import torch
from molgpka.utils.ionization_group import get_ionization_aid
from molgpka.utils.descriptor import mol2vec
from molgpka.utils.net import GCNNet

root = osp.abspath(osp.dirname(__file__))


def load_model(model_file, device="cpu"):
    model = GCNNet().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model


def model_pred(m2, aid, model, device="cpu"):
    data = mol2vec(m2, aid)
    with torch.no_grad():
        data = data.to(device)
        pKa = model(data)
        pKa = pKa.cpu().numpy()
        pka = pKa[0][0]
    return pka


def predict_acid(mol):
    model_file = osp.join(root, "models/weight_acid.pth")
    model_acid = load_model(model_file)

    acid_idxs = get_ionization_aid(mol, acid_or_base="acid")
    acid_res = {}
    for aid in acid_idxs:
        apka = model_pred(mol, aid, model_acid)
        acid_res.update({aid: apka})
    return acid_res


def predict_base(mol):
    model_file = osp.join(root, "models/weight_base.pth")
    model_base = load_model(model_file)

    base_idxs = get_ionization_aid(mol, acid_or_base="base")
    base_res = {}
    for aid in base_idxs:
        bpka = model_pred(mol, aid, model_base)
        base_res.update({aid: bpka})
    return base_res


def predict_pka(mol, uncharged=True):
    if uncharged:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    mol = AllChem.AddHs(mol)
    base_dict = predict_base(mol)
    acid_dict = predict_acid(mol)
    return dict(base=base_dict, acid=acid_dict)


def remote_predict_pka(smi):
    upload_url = r"http://xundrug.cn:5001/modules/upload0/"
    param = {"Smiles": ("tmg", smi)}
    headers = {"token": "O05DriqqQLlry9kmpCwms2IJLC0MuLQ7"}
    response = requests.post(url=upload_url, files=param, headers=headers)
    jsonbool = int(response.headers["ifjson"])
    if jsonbool == 1:
        res_json = response.json()
        if res_json["status"] == 200:
            pka_datas = res_json["gen_datas"]
            return pka_datas
        else:
            raise RuntimeError("Error for pKa prediction")
    else:
        raise RuntimeError("Error for pKa prediction")


if __name__ == "__main__":
    mol = Chem.MolFromSmiles("CN(C)CCCN1C2=CC=CC=C2SC2=C1C=C(C=C2)C(C)=O")
    base_dict, acid_dict = predict_pka(mol)
    print("base:", base_dict["base"])
    print("acid:", acid_dict["acid"])
