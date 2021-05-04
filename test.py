import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json
import pdb

from utils import AverageMeter, calculate_accuracy, calculate_precision, calculate_recall


def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=2)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })
    test_results['results'][video_id] = video_results


def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    
    total_output_buffer = []
    total_target_buffer = []

    for i, (inputs, positions, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        
       
        with torch.no_grad():
            inputs = Variable(inputs)
            positions = Variable(positions)
            targets = Variable(targets).cuda()
            outputs = model(inputs, positions)

            if not opt.no_softmax_in_test:
                outputs = F.softmax(outputs, dim=1)
            
            total_output_buffer.append(outputs)
            total_target_buffer.append(targets)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j].item() != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j].item()

        if (i % 100) == 0:
            with open(
                    os.path.join(opt.result_path, '{}.json'.format(
                        opt.test_subset)), 'w') as f:
                json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))

    total_output = torch.cat(total_output_buffer, dim=0)
    total_target = torch.cat(total_target_buffer, dim=0)

    acc = calculate_accuracy(total_output, total_target)
    precision = calculate_precision(total_output, total_target)
    recall = calculate_recall(total_output,total_target)

    print("Overall:\t acc: {}, precision: {}, recall: {}".format(
        acc, precision, recall
    ))
    with open(
            os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
            'w') as f:
        json.dump(test_results, f)
