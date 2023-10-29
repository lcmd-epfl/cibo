from joblib import Parallel, delayed
import morfeus
from morfeus.conformer import ConformerEnsemble
from morfeus import SASA, Dispersion
import numpy as np
import pdb
import tqdm

def get_morfeus_desc(smiles):

  ce = ConformerEnsemble.from_rdkit(smiles)

  ce.prune_rmsd()

  ce.sort()

  for conformer in ce:
    sasa = SASA(ce.elements, conformer.coordinates)
    disp = Dispersion(ce.elements, conformer.coordinates)
    conformer.properties["sasa"] = sasa.area
    conformer.properties["p_int"] = disp.p_int
    conformer.properties["p_min"] = disp.p_min
    conformer.properties["p_max"] = disp.p_max


  ce.get_properties()
  a= ce.boltzmann_statistic("sasa")
  b= ce.boltzmann_statistic("p_int")
  c= ce.boltzmann_statistic("p_min")
  d = ce.boltzmann_statistic("p_max")

  return np.array([a,b,c,d])



dataset_url = "https://raw.githubusercontent.com/doylelab/rxnpredict/master/data_table.csv"
#load url directly into pandas dataframe
import pandas as pd
data = pd.read_csv(dataset_url) #.fillna({"base_smiles":"","ligand_smiles":"","aryl_halide_number":0,"aryl_halide_smiles":"","additive_number":0, "additive_smiles": ""}, inplace=False)
# remove rows with nan
data = data.dropna()


ligands = np.unique(data["ligand_smiles"].values)


def get_morfeus_desc_for_ligand(ligand):
    try:
        des = get_morfeus_desc(ligand)
        return des
    except Exception as e:
        return np.nan



results = Parallel(n_jobs=4)(delayed(get_morfeus_desc_for_ligand)(ligand)
                              for ligand in ligands)

# Now 'results' will hold the descriptors for each ligand, or an exception message if something went wrong.
results = np.array(results)

pdb.set_trace()