#python -u run.py --gpuid 0 --dataset BBBP --exp_name final --batch_size 32 --lr 1e-5 --optimizer ADAM --epochs 200 \
#                 --grad_clip 1.0 --patience 50 --scaffold_id 1


python -u run.py --gpuid 0 --dataset BBBP --exp_name test1 --batch_size 32 --lr 1e-5 --optimizer ADAM --epochs 200 \
                 --grad_clip 1.0 --patience 50 --scaffold_id 1

python -u run.py --gpuid 0 --dataset BBBP --exp_name test2 --batch_size 32 --lr 1e-5 --optimizer ADAM --epochs 200 \
                 --grad_clip 1.0 --patience 50 --scaffold_id 2

python -u run.py --gpuid 0 --dataset BBBP --exp_name test3 --batch_size 32 --lr 1e-5 --optimizer ADAM --epochs 200 \
                 --grad_clip 1.0 --patience 50 --scaffold_id 3

python -u run.py --gpuid 0 --dataset BBBP --exp_name test4 --batch_size 32 --lr 1e-5 --optimizer ADAM --epochs 200 \
                 --grad_clip 1.0 --patience 50 --scaffold_id 4

python -u run.py --gpuid 0 --dataset BBBP --exp_name test5 --batch_size 32 --lr 1e-5 --optimizer ADAM --epochs 200 \
                 --grad_clip 1.0 --patience 50 --scaffold_id 5