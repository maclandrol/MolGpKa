import numpy as np
import click

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from rdkit.Chem import rdmolops
from multiprocessing import Pool
import joblib
from sklearn.model_selection import train_test_split
from molgpka.utils.descriptor import mol2vec
from molgpka.graph_model import GraphModelTrainer
from molgpka.ap_model import read_datasets, gen_data, train

APLength = 6


def generate_graph_datasets(mols):
    datasets = []
    for mol in mols:
        if not mol:
            continue
        atom_idx = int(mol.GetProp("idx"))
        pka = float(mol.GetProp("pka"))
        data = mol2vec(mol, atom_idx, evaluation=False, pka=pka)
        datasets.append(data)

    train_dataset, valid_dataset = train_test_split(datasets, test_size=0.1)
    data_dict = dict(train=train_dataset, valid=valid_dataset)
    return data_dict


def calc_ap(mol):
    aid = int(mol.GetProp("idx"))
    pka = float(mol.GetProp("pka"))
    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(
        mol, maxLength=APLength, fromAtoms=[aid]
    )
    arr = np.zeros(
        1,
    )
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr, pka


def generate_datasets(mols, outfile):
    fps, targets = [], []
    for m in mols:
        if not m:
            continue
        fp, pka = calc_ap(m)
        fps.append(fp)
        targets.append(pka)
    fps = np.asarray(fps)
    targets = np.asarray(targets)
    np.savez_compressed(outfile, fp=fps, pka=targets)
    return


@click.group()
def cli():
    pass


@cli.command()
@click.option("--data", help="Path to input dataset")
@click.option("--epochs", type=int, default=50, help="Number of epochs")
@click.option(
    "--output",
    default="models/weight_ap_{epoch}.pth",
    help="Format of the output path add '{epoch}' in the string to save every epochs",
)
def train_ap(data, epochs, output):
    fps, pkas = read_datasets(data)
    train_loader, valid_loader = gen_data(fps, pkas)
    train(train_loader, valid_loader, epochs=epochs, output=output)


@cli.command()
@click.option("--data", help="Path to input dataset")
@click.option("--epochs", type=int, default=50, help="Number of epochs")
@click.option("--batch-size", type=int, default=128, help="Batch size")
@click.option("--lr", type=float, default=0.0001, help="Learning rate")
@click.option(
    "--output",
    default="models/weight_ap_{epoch}.pth",
    help="Format of the output path add '{epoch}' in the string to save every epochs",
)
def train_graph(data, epochs, batch_size, lr, output):
    graph_trainer = GraphModelTrainer(data, batch_size=batch_size, lr=lr)
    graph_trainer.run(epochs=epochs, output_path=output)


@cli.command()
@click.option("--infile", required=True, help="path to input dataset in sdf format")
@click.option("--outfile", help="path to output dataset in npz or pickle")
@click.option(
    "--graph",
    is_flag=True,
    help="whether data preparation if for the baseline or graph",
)
def prepare_dataset(infile, outfile, graph):
    mols = Chem.SDMolSupplier(infile, removeHs=False)

    if not graph:
        if not outfile:
            outfile = infile + ".npz"
        generate_datasets(mols, outfile)
    else:
        dataset = generate_graph_datasets(mols)
        if not outfile:
            outfile = infile + ".pkl"
        joblib.dump(dataset, outfile)


def main_cli():
    return cli()


if __name__ == "__main__":
    cli()
