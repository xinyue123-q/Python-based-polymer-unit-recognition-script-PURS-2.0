from pathlib import Path
from typing import Tuple

import torch
import pandas as pd
from pandas.api.types import CategoricalDtype

from rdkit import Chem

import torchgraphs as tg
import PURS as pu

symbols = CategoricalDtype([
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
    'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
    'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
    'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
    'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
], ordered=True)

bonds = CategoricalDtype([
    'SINGLE',
    'DOUBLE',
    'TRIPLE',
    'AROMATIC'
], ordered=True)

def get_pu_dist(file_name):
 
    smi_list,name_list0,name_list,mol_list,num_list=pu.process_smiles(file_name)
    ring_total_list,total_neighbor_data,total_inner_dist,total_end_atom_pair=pu.get_pu(smi_list,name_list)
    total_bratch_dist = pu.get_bratch_dist2(smi_list,name_list)
    total_neighbor_data,total_inner_dist,ring_total_list = pu.update_bratch(name_list,smi_list,total_neighbor_data,total_inner_dist,total_bratch_dist)
    total_neighbor_data,total_inner_dist,ring_total_list = pu.get_new_neighbor_data(total_neighbor_data,total_inner_dist,name_list,smi_list)
    matrix_list=pu.get_adj(ring_total_list,total_neighbor_data,name_list)
    pu_index = pu.get_pu_dict(total_neighbor_data,ring_total_list)
    pair_atom_dist = pu.get_pair_atom(total_neighbor_data,total_inner_dist)
    MACCS_dict = pu.get_MACCS(ring_total_list,False)
    #senders_list,receivers_list,edge_num_list = pu.edge_index(matrix_list)
    #node_index, node_num_list = pu.get_node_index(total_neighbor_data,ring_total_list)
    edge_list,edge_num_list,senders_list,receivers_list,node_feature_dist=pu.get_feature(smi_list,name_list,matrix_list,pu_index,pair_atom_dist,total_inner_dist,MACCS_dict)
    #node_feature_dist = pu.get_node_feature(name_list,MACCS_dict,pu_index,node_num_list)
    graph_dist = {}
    for idx, name in enumerate(name_list):
      
        
        #num_nodes = node_num_list[idx]
        num_edges = edge_num_list[idx]
        node_features = node_feature_dist[name]
        num_nodes = len(node_features)
        edge_features = edge_list[idx]
        senders = senders_list[name]
        receivers = receivers_list[name] 
        
        #print("len(node_features)")
        #print(len(node_features))
        #print("senders")
        #print(senders)
        #print("receivers")
        #print(receivers)
        graph=tg.Graph(
        num_nodes=num_nodes,
        num_edges=num_edges,
        node_features=node_features,
        edge_features=edge_features,
        senders=senders,
        receivers=receivers)
        
        graph_dist[name]=graph
    
  
    return graph_dist
  
#def smiles_to_graph(smiles: str) -> tg.Graph:
   # molecule = Chem.MolFromSmiles(smiles)

    #atoms_df = []
    #for i in range(molecule.GetNumAtoms()):
       # atom = molecule.GetAtomWithIdx(i)
       # atoms_df.append({
            #'index': i,
            #'symbol': atom.GetSymbol(),
            #'degree': atom.GetDegree(),
            #'hydrogens': atom.GetTotalNumHs(),
            #'impl_valence': atom.GetImplicitValence(),
        #})
    #atoms_df = pd.DataFrame.from_records(atoms_df, index='index',
                                         #columns=['index', 'symbol', 'degree', 'hydrogens', 'impl_valence'])
    #atoms_df.symbol = atoms_df.symbol.astype(symbols)

    #node_features = torch.tensor(pd.get_dummies(atoms_df, columns=['symbol']).values, dtype=torch.float)

    #bonds_df = []
    #for bond in molecule.GetBonds():
        #bonds_df.append({
            #'sender': bond.GetBeginAtomIdx(),
            #'receiver': bond.GetEndAtomIdx(),
            #'type': bond.GetBondType().name,
            #'conj': bond.GetIsConjugated(),
            #'ring': bond.IsInRing()
        #})
        #bonds_df.append({
            #'sender': bond.GetEndAtomIdx(),
            #'receiver': bond.GetBeginAtomIdx(),
            #'type': bond.GetBondType().name,
            #'conj': bond.GetIsConjugated(),
            #'ring': bond.IsInRing()
        #})
    #bonds_df = pd.DataFrame.from_records(bonds_df, columns=['sender', 'receiver', 'type', 'conj', 'ring'])\
        #.set_index(['sender', 'receiver'])
    #bonds_df.conj = bonds_df.conj * 2. - 1
    #bonds_df.ring = bonds_df.ring * 2. - 1
    #bonds_df.type = bonds_df.type.astype(bonds)

    #edge_features = torch.tensor(pd.get_dummies(bonds_df, columns=['type']).values.astype(float), dtype=torch.float)
    #senders = torch.tensor(bonds_df.index.get_level_values('sender'), dtype=torch.long)
    #receivers = torch.tensor(bonds_df.index.get_level_values('receiver'), dtype=torch.long)

    
    #return tg.Graph(
        #num_nodes=molecule.GetNumAtoms(),
        #num_edges=molecule.GetNumBonds() * 2,
        #node_features=node_features,
        #edge_features=edge_features,
        #senders=senders,
        #receivers=receivers
    #)

def smiles_to_graph(graph_dist,name):
    name = str(name)
    graph = graph_dist[name]
    return graph
   

class SolubilityDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.graph_dist = get_pu_dist(path)
        self.df = pd.read_csv(path)
        # self.df['molecules'] = self.df.smiles.apply(smiles_to_graph)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item) -> Tuple[tg.Graph, float]:
     
        mol = smiles_to_graph(self.graph_dist,self.df['Compound ID'].iloc[item])
        target = self.df['PCE_max'].iloc[item]
        return mol, torch.tensor(target)


def describe(cfg):
    pd.options.display.precision = 2
    pd.options.display.max_columns = 999
    pd.options.display.expand_frame_repr = False
    target = Path(cfg.target).expanduser().resolve()
    if target.is_dir():
        paths = target.glob('*.pt')
    else:
        paths = [target]
    for p in paths:
        print(f"Loading dataset from: {p}")
        dataset = SolubilityDataset(p)
        print(f"{p.with_suffix('').name.capitalize()} contains:\n"
              f"{dataset.df.drop(columns=['molecules']).describe().transpose()}")


def main():
    from argparse import ArgumentParser
    from config import Config

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    sp_print = subparsers.add_parser('print', help='Print parsed configuration')
    sp_print.add_argument('config', nargs='*')
    sp_print.set_defaults(command=lambda c: print(c.toYAML()))

    sp_describe = subparsers.add_parser('describe', help='Describe existing datasets')
    sp_describe.add_argument('config', nargs='*')
    sp_describe.set_defaults(command=describe)

    args = parser.parse_args()
    cfg = Config.build(*args.config)
    args.command(cfg)


if __name__ == '__main__':
    main()
