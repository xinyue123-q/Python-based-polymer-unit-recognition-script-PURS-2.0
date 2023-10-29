# Python-based-polymer-unit-recognition-script-PURS-2.0
1.High throughput identification of polymer-units(repeating units) from SMILES codes of polymers

2.GNN models based on polymer-unit - Model acceleration and interpretative enhancement

### Installation
#### 1.Use only the recognition function
```
conda create -n PURS
conda activate PURS
pip install -r requirements.yml
```
#### 2.Interpretable GNNS based on polymer-units--PU-gn-exp

gn-exp is developed as the baseline model.Please follow the link "https://github.com/baldassarreFe/graph-network-explainability" to install gn-exp first and then refer to Readme in the PU-gn-exp folder.

#### 3.Prediction model based on polymer-unit--PU-MPNN
mol-MPNN is developed as the baseline model.Please follow the link "https://github.com/seokhokang/mol_mpnn" to install mol-MPNN first and then refer to Readme in the PU-MPNN folder.


