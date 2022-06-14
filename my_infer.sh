#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate py310

runid="-`date -Iseconds`"

runname="chika$runid"
git add --all
#git commit -m "$runname"

#screen -S $runname 
#checkpoint="./saved_models/resu2-2022-01-28T22:29:21+08:00/checkpoints/itr_-1"
checkpoint="./saved_models/resu2-2022-05-01T00:01:26+08:00/checkpoints/itr_-1"
#checkpoint="./saved_models/resu2-2022-02-19T21:02:02+08:00/checkpoints/itr_-1"

mkdir "./saved_models/resu2/images/$runname"

rlaunch --cpu=4 --gpu=4 --memory=41920 -- python -m torch.distributed.launch --nproc_per_node=4 -r 3 -t 3 --log_dir ./logs/$runid train.py --mode=test --world_size=4 --dataloaders=8 --test_input_poses_images=./test_data/output --test_input_person_images=./test_data/test_images_draw3 --test_output_dir=./saved_models/resu2/images/$runname --test_checkpoint_dir=$checkpoint   --shader_pose_use_parser_udp_test=False --dataloader_imgsize=256   --shader_clip_message=False --test_rnn_iterate_on_last_frames=0 

conda activate base
echo generating video from images...
cd ./saved_models/resu2/images/$runname
ffmpeg  -f image2 -i ./%d.png -s 1024x576 -c:v libx264 -pix_fmt yuv420p ../$runname.mp4 -q:v 0 -q:a 0 -crf 1