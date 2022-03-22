import argparse
import os
import time
from datetime import datetime
from distutils.util import strtobool

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader import (FileDataset,
                         RandomResizedCropWithAutoCenteringAndZeroPadding)
from conr import CoNR


def save_output(image_name, inputs_v, d_dir=".", crop=None):
    import cv2

    inputs_v = inputs_v.detach().squeeze()
    input_np = torch.clamp(inputs_v*255, 0, 255).byte().cpu().numpy().transpose(
        (1, 2, 0))
    # cv2.setNumThreads(1)
    out_render_scale = cv2.cvtColor(input_np, cv2.COLOR_RGBA2BGRA)
    if crop is not None:
        crop = crop.cpu().numpy()[0]
        output_img = np.zeros((crop[0], crop[1], 4), dtype=np.uint8)
        before_resize_scale = cv2.resize(
            out_render_scale, (crop[5]-crop[4]+crop[8]+crop[9], crop[3]-crop[2]+crop[6]+crop[7]), interpolation=cv2.INTER_AREA)  # w,h
        output_img[crop[2]:crop[3], crop[4]:crop[5]] = before_resize_scale[crop[6]:before_resize_scale.shape[0] -
                                                                           crop[7], crop[8]:before_resize_scale.shape[1]-crop[9]]
    else:
        output_img = out_render_scale
    cv2.imwrite(d_dir+"/"+image_name.split(os.sep)[-1]+'.png',
                output_img
                )


def test():
    source_names_list = []
    for name in os.listdir(args.test_input_person_images):
        thissource = os.path.join(args.test_input_person_images, name)
        if os.path.isfile(thissource):
            source_names_list.append([thissource])
        if os.path.isdir(thissource):
            toadd = [os.path.join(thissource, this_file)
                     for this_file in os.listdir(thissource)]
            if (toadd != []):
                source_names_list.append(toadd)
            else:
                print("skipping empty folder :"+thissource)
    image_names_list = []

    for eachlist in source_names_list:
        for name in sorted(os.listdir(args.test_input_poses_images)):
            thistarget = os.path.join(args.test_input_poses_images, name)
            if os.path.isfile(thistarget):
                image_names_list.append([thistarget, *eachlist])
            if os.path.isdir(thistarget):
                print("skipping folder :"+thistarget)

    print("---building models...")
    conrmodel = CoNR(args)
    conrmodel.load_model(path=args.test_checkpoint_dir)
    conrmodel.dist()
    infer(args, conrmodel, image_names_list)


def infer(args, humanflowmodel, image_names_list):
    print("---")
    print("test images: ", len(image_names_list))
    print("---")
    test_salobj_dataset = FileDataset(image_names_list=image_names_list,
                                      fg_img_lbl_transform=transforms.Compose([
                                          RandomResizedCropWithAutoCenteringAndZeroPadding(
                                              (args.dataloader_imgsize, args.dataloader_imgsize), scale=(1, 1), ratio=(1.0, 1.0), center_jitter=(0.0, 0.0)
                                          )]),
                                      shader_pose_use_gt_udp_test=not args.shader_pose_use_parser_udp_test,
                                      shader_target_use_gt_rgb_debug=False
                                      )
    train_data = DataLoader(test_salobj_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.dataloaders)

    # start testing

    train_num = train_data.__len__()
    time_stamp = time.time()
    prev_frame_rgb = []
    prev_frame_a = []
    for i, data in enumerate(train_data):
        data_time_interval = time.time() - time_stamp
        time_stamp = time.time()
        with torch.no_grad():

            data["character_images"] = torch.cat(
                [data["character_images"], *prev_frame_rgb], dim=1)
            data["character_masks"] = torch.cat(
                [data["character_masks"], *prev_frame_a], dim=1)
            data = humanflowmodel.data_norm_image(data)
            pred = humanflowmodel.model_step(data, training=False)

        train_time_interval = time.time() - time_stamp
        time_stamp = time.time()
        print("[infer batch: %5d/%5d] time:%2f+%2f" % (
            i, train_num,
            data_time_interval, train_time_interval
        ))
        with torch.no_grad():

            if args.test_output_video:
                pred_img = pred["shader"]["y_weighted_warp_decoded_rgba"]
                save_output(
                    str(int(data["imidx"].cpu().item())), pred_img, args.test_output_dir, crop=data["pose_crop"])
                if args.test_rnn_iterate_on_last_frames:
                    prev_frame = torch.clamp(
                        pred_img.detach()*255, 0, 255).unsqueeze(0).cpu()
                    prev_frame_rgb.append(prev_frame[:, :, :3, :, :])
                    prev_frame_rgb = prev_frame_rgb[-1 *
                                                    args.test_rnn_iterate_on_last_frames:]
                    prev_frame_a.append(prev_frame[:, :, 3:4, :, :])
                    prev_frame_a = prev_frame_a[-1 *
                                                args.test_rnn_iterate_on_last_frames:]
            if args.test_output_udp:
                pred_img = pred["shader"]["x_target_sudp_a"]
                save_output(
                    "udp_"+str(int(data["imidx"].cpu().item())), pred_img, args.test_output_dir)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument(
        '--runid', default=datetime.now().strftime("-%Y%m%d-%H-%M-%S"))
    parser.add_argument('--pin_memory', type=strtobool, default=True)
    parser.add_argument('--non_blocking', type=strtobool, default=True)
    parser.add_argument('--broadcast_buffers_resnet',
                        type=strtobool, default=False) 
    parser.add_argument('--broadcast_buffers_shader',
                        type=strtobool, default=True)
    parser.add_argument('--shader_pose_use_parser_udp_test',
                        type=strtobool, default=True)
    parser.add_argument('--shader_pose_use_parser_mask_test',
                        type=strtobool, default=False)
    parser.add_argument('--shader_character_use_parser_mask_test',
                        type=strtobool, default=False)

    parser.add_argument('--dataloader_imgsize', type=int, default=128)
    parser.add_argument('--use_amp', type=strtobool, default=False)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_name', default='resu2')
    parser.add_argument('--dataloaders', type=int, default=8)
    parser.add_argument('--mode', default="train")

    parser.add_argument('--test_input_person_images',
                        type=str, default="./test_data/test_images/")
    parser.add_argument('--test_input_poses_images', type=str,
                        default="./test_data/test_poses_images/")
    parser.add_argument('--test_checkpoint_dir', type=str,
                        default=None)
    parser.add_argument('--test_output_dir', type=str,
                        default="./saved_models/resu2/images/test/")
    parser.add_argument('--shader_clip_message', type=strtobool, default=True)
    parser.add_argument('--test_rnn_iterate_on_last_frames',
                        type=int, default=0)
    parser.add_argument('--test_output_video', type=strtobool, default=True)
    parser.add_argument('--test_output_udp', type=strtobool, default=False)

    args = parser.parse_args()

    args.distributed = (args.world_size > 1)
    print("batch_size:", args.batch_size, flush=True)
    if args.distributed:
        print("world_size: ", args.world_size)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", world_size=args.world_size)
    else:
        args.local_rank = 0
    print("local_rank: ", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    torch.backends.cudnn.benchmark = True
    return args


if __name__ == "__main__":
    args = build_args()
    test()
