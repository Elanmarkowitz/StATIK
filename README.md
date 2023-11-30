# StATIK: Structure and Text for Inductive Knowledge Graph Completion

[Paper here](StATIK_NAACL_camera_ready.pdf)

# KGCompletionGNN

```
export ROOT_DATA_DIR=/path/to/base/directory 
```

```
DATASET=FB15k-237 python run_data_processing.py
```
or choose dataset from: `[WN15RR, wikikg90m_kddcup2021, Wikidata5M, FB15k-237]`


To run on multiple GPUs on a single node:

```
torchrun --nproc_per_node=<#GPUs> --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=<PORT#> train.py --root_data_dir=$ROOT_DATA_DIR --train_arg1 --train_arg2 .... --train_argN
```

E.g.
```
torchrun --nproc_per_node=<#GPUs> --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=<PORT#> train.py --root_data_dir=$ROOT_DATA_DIR --dataset=FB15k-237
```

# Citation

```
@inproceedings{Markowitz2022StATIKSA,
  title={StATIK: Structure and Text for Inductive Knowledge Graph Completion},
  author={Elan Markowitz and Keshav Balasubramanian and Mehrnoosh Mirtaheri and Murali Annavaram and A. G. Galstyan and Greg Ver Steeg},
  booktitle={NAACL-HLT},
  year={2022},
  url={https://api.semanticscholar.org/CorpusID:250520198}
}
```