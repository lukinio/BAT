dataset=BBBP
repeat=3

######################################### DEFAULT #########################################
exp_name=default

for i in {0..5}
do
    python -u run.py \
           --gpuid 0 --dataset ${dataset} --repeat ${repeat} --exp_name ${exp_name}_scaffold_${i} --scaffold_id ${i} \
           --batch_size 64 --lr 1e-5 --optimizer ADAM --epochs 200 --grad_clip 1 --patience 50 \
           --d_model 1024 --N 8 --h 16 --N_dense 1 --dropout 0.0 \
           --lambda_attention 0.33 --lambda_distance 0.33 --leaky_relu_slope 0.1 \
           --dense_output_nonlinearity relu --distance_matrix_kernel exp --aggregation_type mean
done
