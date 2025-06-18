# GNN-RE intergration

## Usage

1. **Clone the GNN-RE repository**

 ```bash
   git clone https://github.com/DfX-NYUAD/GNN-RE
   cd GNN-RE
   # follow the original repo’s INSTALL / README instructions here
```
   

2. Copy the integration assets

**Dataset-side files**

```bash
    cp  path/to/CirSTAG/integration/GNN-RE/setup.py  .
    cp  path/to/CirSTAG/integration/GNN-RE/graph_parser_cirstag.py  .
    cp  path/to/CirSTAG/integration/GNN-RE/graph_parser_directed_full_cirstag.py  .
    cp  path/to/CirSTAG/integration/GNN-RE/netlist_to_graph_re_cirstag.pl .
    cp  path/to/CirSTAG/integration/GNN-RE/theCircuit_cirstag.pm .
```

**GNN-side files**
```bash
    cp -r path/to/CirSTAG/cirstag/my_utils ./GraphSAINT/graphsaint/pytorch_version/my_utils
    cp path/to/CirSTAG/integration/GNN-RE/saved_model_2024-04-08 23-39-59,pkl ./GraphSAINT/pytorch_models/
    cp path/to/CirSTAG/integration/GNN-RE/run_me.py ./GraphSAINT/graphsaint/pytorch_version/
    cp path/to/CirSTAG/integration/GNN-RE/apply_perturbation.py ./GraphSAINT/graphsaint/pytorch_version/
```

3. Run the experiment script

```bash
    # ── Dataset setup ────────────────────────────────────────────────
    python3 setup.py
    # Parsed graph datasets will appear under ./Netlist_to_graph/

    # ── CirSTAG evaluation (example: Test_add_mul_sub_16_bit_Syn_65nm) ──
    cd GraphSAINT
    python3 -m graphsaint.pytorch_version.run_me_SAINT --data_prefix ../Netlist_to_graph/Graphs_datasets/Test_add_mul_sub_16_bit_Syn_65nm --train_config ../TCAD.yml --gpu 0
    python3 -m graphsaint.pytorch_version.apply_perturbation --data_prefix ../Netlist_to_graph/Graphs_datasets/Test_add_mul_sub_16_bit_Syn_65nm --train_config ../TCAD.yml --gpu 0

```
