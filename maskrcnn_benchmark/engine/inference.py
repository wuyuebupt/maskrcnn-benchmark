# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize

import cv2
import numpy as np

def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")


    num_ind = 24
    ### good
    ## 5
    ## 7
    ## 8
    ## 23 
    ## 24 
    ## 25
    for i, batch in enumerate(tqdm(data_loader)):
        if i != num_ind:
            print (i, num_ind)
            continue
        # print (i, batch)
        images, targets, image_ids, path = batch
        print (path)
        images = images.to(device)
        print (images)
        print (images.tensors.shape)
        # print (images.image_sizes)
        # print (targets)
        # print (len(targets))
        # print (targets[0].bbox)
        # print (targets[0].extra_fields)
        print (image_ids)

        mean = [102.9801, 115.9465, 122.7717]
        # img_save = img_save.add_(mean[:, None, None])
        # img_save = images_save.cpu().numpy()
        img_save = images.tensors.cpu().numpy()
        # img_save = img_save.view(img_save.shape(1), img_save.shape(2), img_save.shape(3))
        img_save = img_save[0,:,:,:] # .view(img_save.shape(1), img_save.shape(2), img_save.shape(3))
        img_save = np.transpose(img_save, (1, 2, 0))
        print (img_save.shape)
        img_save = img_save + mean
        print (img_save.shape)
        # print (img_save)
        cv2.imwrite('attention/img.jpg', img_save)
        # exit()
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        print (output)
        print (output[0].bbox)
        img_pred = output[0].bbox.cpu().numpy()
        savefile = 'attention/prediction.bin'

        fid = open(savefile, 'wb')
        img_pred.tofile(fid)

        exit()
        print (output[0].extra_fields)
        print (len(output[0].extra_fields['scores']))
        print (len(output[0].extra_fields['labels']))
        # for o in output:
        #     print (o)
        #     exit()

        ## check images visually
        # img_show = images.tensors.cpu().numpy()
        # cv2.imshow('abc', img_show)
        # cv2.waitKey()
      
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
        if i == num_ind:
            break
    print (results_dict)
    exit()
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    start_time = time.time()
    predictions = compute_on_dataset(model, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
