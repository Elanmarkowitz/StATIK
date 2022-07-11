# StATIK: Structure and Text for Inductive Knowledge Graph Completion

[Paper here](StATIK_NAACL_camera_ready.pdf)

# KGCompletionGNN

To run on multiple GPUs on a single node:

 python3 -m torch.distributed.launch --nproc_per_node=<#GPUs> --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=<PORT#> train.py --cmdlinearg1 --cmdlinearg2 .... --cmdlineargN
