python 22_fuse.py --dataset rdb --k 100 --train_start 0 --train_end 10 --valid_start 11 --valid_end 12 \
    --log rdb_22fuse_gcn --gid 2 --embed_dim 300 --edge_gnn gcn

python 22_fuse.py --dataset rdb --k 100 --train_start 0 --train_end 10 --valid_start 11 --valid_end 12 \
    --log rdb_22fuse_gat --gid 2 --embed_dim 300 --edge_gnn gat

python 01_gnn.py --dataset rdb --k 100 --train_start 0 --train_end 10 --valid_start 11 --valid_end 12 \
    --log rdb_gcn --gid 2 --embed_dim 300 --edge_gnn gcn

python 01_gnn.py --dataset rdb --k 100 --train_start 0 --train_end 10 --valid_start 11 --valid_end 12 \
    --log rdb_gat --gid 2 --embed_dim 300 --edge_gnn gat



python 22_fuse.py --dataset rdt --k 100 --train_start 0 --train_end 10 --valid_start 11 --valid_end 12 \
    --log rdt_22fuse_gcn --gid 2 --embed_dim 300 --edge_gnn gcn

python 22_fuse.py --dataset rdt --k 100 --train_start 0 --train_end 10 --valid_start 11 --valid_end 12 \
    --log rdt_22fuse_gat --gid 2 --embed_dim 300 --edge_gnn gat

python 01_gnn.py --dataset rdt --k 100 --train_start 0 --train_end 10 --valid_start 11 --valid_end 12 \
    --log rdt_gcn --gid 2 --embed_dim 300 --edge_gnn gcn

python 01_gnn.py --dataset rdt --k 100 --train_start 0 --train_end 10 --valid_start 11 --valid_end 12 \
    --log rdt_gat --gid 2 --embed_dim 300 --edge_gnn gat