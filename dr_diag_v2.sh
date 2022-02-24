#model_name=dual_path_resnet50_BN_b64_hongyu_diag_largeaug_clrs_asynl_v2
model_name=resnet50_BN_b128_hongyu_lesion_largeaug_clrs
    model_dir=./work_dirs/
    epoch=latest.pth
    dataset=v2drhx

if [ "$1" == "train" ]
then

    CUDA_VISIBLE_DEVICES=4,5,6,7 tools/dist_train.sh configs/$dataset/$model_name.py 4 \
    --gpus 4 --work-dir $model_dir/$model_name/ 
    # --resume-from $model_dir/$model_name/$epoch


elif [ "$1" == "test" ]
then
    CUDA_VISIBLE_DEVICES=0,1,2,3 tools/dist_test.sh configs/$dataset/$model_name.py \
		$model_dir/$model_name/$epoch 4 \
		--out result.json --metric 'auc_multi_cls' 


fi
!
