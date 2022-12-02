# bco
python 22_fuse.py --dataset bco --k 100 --train_start 0 --train_end 11 --valid_start 11 --valid_end 12 \
    --gid 0 --log bco_22fuse_gcn --edge_gnn gcn

python 22_fuse.py --dataset bco --k 100 --train_start 0 --train_end 11 --valid_start 11 --valid_end 12 \
    --gid 0 --log bco_22fuse_gat --edge_gnn gat

python 01_gnn.py --dataset bco --k 100 --train_start 0 --train_end 11 --valid_start 11 --valid_end 12 \
    --gid 0 --log bco_gcn --edge_gnn gcn

python 01_gnn.py --dataset bco --k 100 --train_start 0 --train_end 11 --valid_start 11 --valid_end 12 \
    --gid 0 --log bco_gat --edge_gnn gat


# bca
python 22_fuse.py --dataset bca --k 100 --train_start 0 --train_end 11 --valid_start 11 --valid_end 12 \
    --gid 0 --log bca_22fuse_gcn --edge_gnn gcn

python 22_fuse.py --dataset bca --k 100 --train_start 0 --train_end 11 --valid_start 11 --valid_end 12 \
    --gid 0 --log bca_22fuse_gat --edge_gnn gat

python 01_gnn.py --dataset bca --k 100 --train_start 0 --train_end 11 --valid_start 11 --valid_end 12 \
    --gid 0 --log bca_gcn --edge_gnn gcn

python 01_gnn.py --dataset bca --k 100 --train_start 0 --train_end 11 --valid_start 11 --valid_end 12 \
    --gid 0 --log bca_gat --edge_gnn gat