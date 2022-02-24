#model_name=dual_path_resnet18_3d_BN_b8_pretrained_45eps_diag_sidsampler_clrs_depth_asynl_v2
model_name=resnet18_3d_BN_b32_pretrained_45eps_lesion_sidsampler_clrs_depth_v2
model_dir=./work_dirs/
dataset=v2cthx
epoch=latest.pth

if [ "$1" == "train" ]
then
    echo "MISTAKE"
    #CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_train.sh configs/$dataset/$model_name.py 4 \
    #--gpus 4 --work-dir $model_dir/$model_name/
    # --resume-from $model_dir/$model_name/$epoch

    #CUDA_VISIBLE_DEVICES=0 tools/dist_train.sh configs/$dataset/$model_name.py 1 \
    #--gpus 1 --work-dir $model_dir/$model_name/

elif [ "$1" == "trainsid" ]  # training each epoch by sampling only on series from that patient
then

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 tools/dist_train.sh configs/$dataset/$model_name.py 8 \
    --gpus 8 --work-dir $model_dir/$model_name/ --sid_sampler
    # --resume-from $model_dir/$model_name/$epoch


elif [ "$1" == "test" ]
then

	  CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_test.sh configs/$dataset/$model_name.py \
		$model_dir/$model_name/$epoch 4 \
		--out result.json --metric 'auc_multi_cls'

fi

