
dataset=BACE
repeat=3

######################################### DEFAULT #########################################
exp_name=default

# bace_docked.csv
python -u run.py \
       --gpuid 0 --dataset ${dataset} --repeat ${repeat} --exp_name ${exp_name} \
       --batch_size 64 --lr 1e-5 --optimizer ADAM --epochs 200 --grad_clip 1 --patience 50 \
       --d_model 1024 --N 8 --h 16 --N_dense 1 --dropout 0.0 \
       --lambda_attention 0.33 --lambda_distance 0.33 --leaky_relu_slope 0.1 \
       --dense_output_nonlinearity relu --distance_matrix_kernel exp --aggregation_type mean

# bace_poses, .mol2
python -u run.py \
       --gpuid 0 --dataset ${dataset} --mol_files --repeat ${repeat} --exp_name ${exp_name}_mol \
       --batch_size 64 --lr 1e-5 --optimizer ADAM --epochs 200 --grad_clip 1 --patience 50 \
       --d_model 1024 --N 8 --h 16 --N_dense 1 --dropout 0.0 \
       --lambda_attention 0.33 --lambda_distance 0.33 --leaky_relu_slope 0.1 \
       --dense_output_nonlinearity relu --distance_matrix_kernel exp --aggregation_type mean


######################################### SOFTMAX #########################################
exp_name=softmax

# bace_docked.csv
python -u run.py \
       --gpuid 0 --dataset ${dataset} --repeat ${repeat} --exp_name ${exp_name} \
       --batch_size 64 --lr 1e-5 --optimizer ADAM --epochs 200 --grad_clip 1 --patience 50 \
       --d_model 1024 --N 8 --h 16 --N_dense 1 --dropout 0.0 \
       --lambda_attention 0.33 --lambda_distance 0.33 --leaky_relu_slope 0.1 \
       --dense_output_nonlinearity relu --distance_matrix_kernel softmax --aggregation_type mean

# bace_poses, .mol2
python -u run.py \
       --gpuid 0 --dataset ${dataset} --mol_files --repeat ${repeat} --exp_name ${exp_name}_mol \
       --batch_size 64 --lr 1e-5 --optimizer ADAM --epochs 200 --grad_clip 1 --patience 50 \
       --d_model 1024 --N 8 --h 16 --N_dense 1 --dropout 0.0 \
       --lambda_attention 0.33 --lambda_distance 0.33 --leaky_relu_slope 0.1 \
       --dense_output_nonlinearity relu --distance_matrix_kernel softmax --aggregation_type mean


######################################### DUMMY NODE #########################################
exp_name=dummy_node

# bace_docked.csv
python -u run.py \
       --gpuid 0 --dataset ${dataset} --repeat ${repeat} --exp_name ${exp_name} \
       --batch_size 64 --lr 1e-5 --optimizer ADAM --epochs 200 --grad_clip 1 --patience 50 \
       --d_model 1024 --N 8 --h 16 --N_dense 1 --dropout 0.0 \
       --lambda_attention 0.33 --lambda_distance 0.33 --leaky_relu_slope 0.1 \
       --dense_output_nonlinearity relu --distance_matrix_kernel exp --aggregation_type dummy_node

# bace_poses, .mol2
python -u run.py \
       --gpuid 0 --dataset ${dataset} --mol_files --repeat ${repeat} --exp_name ${exp_name}_mol \
       --batch_size 64 --lr 1e-5 --optimizer ADAM --epochs 200 --grad_clip 1 --patience 50 \
       --d_model 1024 --N 8 --h 16 --N_dense 1 --dropout 0.0 \
       --lambda_attention 0.33 --lambda_distance 0.33 --leaky_relu_slope 0.1 \
       --dense_output_nonlinearity relu --distance_matrix_kernel exp --aggregation_type dummy_node


######################################### SOFTMAX - DUMMY NODE #########################################
exp_name=softmax_dummy_node

# bace_docked.csv
python -u run.py \
       --gpuid 0 --dataset ${dataset} --repeat ${repeat} --exp_name ${exp_name} \
       --batch_size 64 --lr 1e-5 --optimizer ADAM --epochs 200 --grad_clip 1 --patience 50 \
       --d_model 1024 --N 8 --h 16 --N_dense 1 --dropout 0.0 \
       --lambda_attention 0.33 --lambda_distance 0.33 --leaky_relu_slope 0.1 \
       --dense_output_nonlinearity relu --distance_matrix_kernel softmax --aggregation_type dummy_node

# bace_poses, .mol2
python -u run.py \
       --gpuid 0 --dataset ${dataset} --mol_files --repeat ${repeat} --exp_name ${exp_name}_mol \
       --batch_size 64 --lr 1e-5 --optimizer ADAM --epochs 200 --grad_clip 1 --patience 50 \
       --d_model 1024 --N 8 --h 16 --N_dense 1 --dropout 0.0 \
       --lambda_attention 0.33 --lambda_distance 0.33 --leaky_relu_slope 0.1 \
       --dense_output_nonlinearity relu --distance_matrix_kernel softmax --aggregation_type dummy_node
