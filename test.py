import os
import time
import csv
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

# from models import ResNet
from metrics import AverageMeter, Result
from dense_to_sparse import UniformSampling, SimulatedStereo
import criteria
import utils


args = utils.parse_command()
print(args)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()

def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    # rgb_traindir = os.path.join('data', args.data, 'RGB_RAINNIGHT', 'val')
    # dep_traindir = os.path.join('data', args.data, 'Depth_RAINNGIHT', 'val')
    # sky_traindir = os.path.join('data', args.data, 'sky_rainnight', 'val')
    rgb_valdir = os.path.join('data', args.data, 'rgb_day', 'test')
    dep_valdir = os.path.join('data', args.data, 'depth_day', 'test')
    sky_valdir = os.path.join('data', args.data, 'sky_day', 'test')
    # rgb_valdir = os.path.join('data', args.data, 'RGB_RAINNIGHT', 'val')
    # dep_valdir = os.path.join('data', args.data, 'Depth_RAINNIGHT', 'val')
    # sky_valdir = os.path.join('data', args.data, 'sky_rainnight', 'val')
    # rgb_valdir = os.path.join('data', args.data, 'RGB_WINTER', 'val')
    # dep_valdir = os.path.join('data', args.data, 'Depth_WINTER', 'val')
    # sky_valdir = os.path.join('data', args.data, 'sky_winter', 'val')
    train_loader = None
    val_loader = None

    # sparsifier is a class for generating random sparse depth input from the ground truth
    sparsifier = None
    max_depth = args.max_depth if args.max_depth >= 0.0 else np.inf
    if args.sparsifier == UniformSampling.name:
        sparsifier = UniformSampling(num_samples=args.num_samples, max_depth=max_depth)
    elif args.sparsifier == SimulatedStereo.name:
        sparsifier = SimulatedStereo(num_samples=args.num_samples, max_depth=max_depth)

    if args.data == 'synthia':
        from Synthia_dataloader import SYNTHIADataset
        # if not args.evaluate:
        #     train_dataset = SYNTHIADataset(rgb_traindir, dep_traindir, sky_traindir, type='train',
        #         modality=args.modality, sparsifier=sparsifier)
        val_dataset = SYNTHIADataset(rgb_valdir, dep_valdir, sky_valdir, type='val',
            modality=args.modality, sparsifier=sparsifier)

    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti or synthia.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    # if not args.evaluate:
    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=args.batch_size, shuffle=True,
    #         num_workers=args.workers, pin_memory=True, sampler=None,
    #         worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader

# def vgg():
#     vgg = net.vgg
#     vgg.load_state_dict(torch.load('vgg_normalised.pth'))
#     vgg = nn.Sequential(*list(vgg.children())[:31])
#     return vgg
#
# def test_transform(size, crop):
#     transform_list = []
#     if size != 0:
#         transform_list.append(transforms.Resize(size))
#     if crop:
#         transform_list.append(transforms.CenterCrop(size))
#     transform_list.append(transforms.ToTensor())
#     transform = transforms.Compose(transform_list)
#     return transform
#
# def make_style():
#     style_dir = Path('style_day')
#     style_paths = [f for f in style_dir.glob('*')]
#     style_tf = test_transform(512, 'store_true')
#     style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
#     return style
#
# def make_style_f():
#     style_f = vgg()(make_style())
#     return style_f

def main():
    global args, best_result, output_directory, train_csv, test_csv


    # style_tf = test_transform(512, 'store_true')
    # style_dir = Path('style_day')
    # style_paths = [f for f in style_dir.glob('*')]

    # evaluation mode
    start_epoch = 0
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no best model found at '{}'".format(args.evaluate)
        print("=> loading best model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        # output_directory = os.path.join('results', 'test')
        # output_directory = os.path.join('RAINNIGHT_test')
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        _, val_loader = create_data_loaders(args)
        args.evaluate = True
        output_directory = utils.get_output_directory(args)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        train_csv = os.path.join(output_directory, 'train.csv')
        test_csv = os.path.join(output_directory, 'test.csv')
        best_txt = os.path.join(output_directory, 'best.txt')
        validate(val_loader, model, checkpoint['epoch'], write_to_file=True)
        return

    # optionally resume from a checkpoint
    # elif args.resume:
    #     chkpt_path = args.resume
    #     assert os.path.isfile(chkpt_path), \
    #         "=> no checkpoint found at '{}'".format(chkpt_path)
    #     print("=> loading checkpoint '{}'".format(chkpt_path))
    #     checkpoint = torch.load(chkpt_path)
    #     args = checkpoint['args']
    #     start_epoch = checkpoint['epoch'] + 1
    #     best_result = checkpoint['best_result']
    #     model = checkpoint['model']
    #     optimizer = checkpoint['optimizer']
    #     output_directory = os.path.dirname(os.path.abspath(chkpt_path))
    #     print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    #     train_loader, val_loader = create_data_loaders(args)
    #     args.resume = True
    #
    # # create new model
    # else:
    #     train_loader, val_loader = create_data_loaders(args)
    #     print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
    #     in_channels = len(args.modality)
    #     if args.arch == 'resnet50':
    #         model = ResNet(layers=50, decoder=args.decoder, output_size=train_loader.dataset.output_size,
    #             in_channels=in_channels, pretrained=args.pretrained)
    #     elif args.arch == 'resnet18':
    #         model = ResNet(layers=18, decoder=args.decoder, output_size=train_loader.dataset.output_size,
    #             in_channels=in_channels, pretrained=args.pretrained)
    #     print("=> model created.")
    #     optimizer = torch.optim.SGD(model.parameters(), args.lr, \
    #         momentum=args.momentum, weight_decay=args.weight_decay)
    #
    #     # model = torch.nn.DataParallel(model).cuda() # for multi-gpu training
    #     model = model.cuda()

    # define loss function (criterion) and optimizer
    if args.criterion == 'l2':
        criterion = criteria.MaskedMSELoss().cuda()
    elif args.criterion == 'l1':
        criterion = criteria.MaskedL1Loss().cuda()

    # create results folder, if not already exists
    # output_directory = utils.get_output_directory(args)
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
    # train_csv = os.path.join(output_directory, 'train.csv')
    # test_csv = os.path.join(output_directory, 'test.csv')
    # best_txt = os.path.join(output_directory, 'best.txt')

    # create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # from torch.utils.tensorboard import SummaryWriter
    #
    # logwriter = SummaryWriter()

    # for epoch in range(start_epoch, args.epochs):
    #     utils.adjust_learning_rate(optimizer, epoch, args.lr)
    #     train(train_loader, model, criterion, optimizer, epoch) # train for one epoch
    #     result, img_merge = validate(val_loader, model, epoch) # evaluate on validation set
    #
    #     # remember best rmse and save checkpoint
    #     is_best = result.rmse < best_result.rmse
    #     if is_best:
    #         best_result = result
    #         with open(best_txt, 'w') as txtfile:
    #             txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
    #                 format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
    #         if img_merge is not None:
    #             img_filename = output_directory + '/comparison_best.png'
    #             utils.save_image(img_merge, img_filename)
    #
    #     # output_directory = os.path.join('RAINNIGHT_test')
    #     utils.save_checkpoint({
    #         'args': args,
    #         'epoch': epoch,
    #         'arch': args.arch,
    #         'model': model,
    #         'best_result': best_result,
    #         'optimizer' : optimizer,
    #     }, is_best, epoch, output_directory)


# def train(train_loader, model, criterion, optimizer, epoch):
#     average_meter = AverageMeter()
#     model.train() # switch to train mode
#     end = time.time()
#     for i, (input, target) in enumerate(train_loader):
#
#         input, target = input.cuda(), target.cuda()
#         torch.cuda.synchronize()
#         data_time = time.time() - end
#
#         # compute pred
#         end = time.time()
#         pred = model(input)
#         loss = criterion(pred, target)
#         optimizer.zero_grad()
#         loss.backward() # compute gradient and do SGD step
#         optimizer.step()
#         torch.cuda.synchronize()
#         gpu_time = time.time() - end
#
#         # measure accuracy and record loss
#         result = Result()
#         result.evaluate(pred.data, target.data)
#         average_meter.update(result, gpu_time, data_time, input.size(0))
#         end = time.time()
#
#         if (i + 1) % args.print_freq == 0:
#             print('=> output: {}'.format(output_directory))
#             print('Train Epoch: {0} [{1}/{2}]\t'
#                   't_Data={data_time:.3f}({average.data_time:.3f}) '
#                   't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
#                   'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
#                   'MAE={result.mae:.2f}({average.mae:.2f}) '
#                   'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
#                   'REL={result.absrel:.3f}({average.absrel:.3f}) '
#                   'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
#                   epoch, i+1, len(train_loader), data_time=data_time,
#                   gpu_time=gpu_time, result=result, average=average_meter.average()))
#
#     avg = average_meter.average()
#     with open(train_csv, 'a') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
#             'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
#             'gpu_time': avg.gpu_time, 'data_time': avg.data_time})


def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        # pred = np.array(pred)
        # pred = np.where(target == 0, 0, pred)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        error_pred = np.squeeze(pred.data.cpu().numpy())
        error_target = np.squeeze(target.cpu().numpy())
        error_pred = np.where(error_target == 0, 0, error_pred) #revised
        error = np.absolute(error_pred - error_target)
        errorname = output_directory + '/error_map_' + str(i) + '.png'
        utils.save_image(error, errorname)
        # save 8 images for visualization
        skip = 10
        if args.modality == 'd':
            img_merge = None
        else:
            if args.modality == 'rgb':
                rgb = input
            elif args.modality == 'rgbd':
                rgb = input[:,:3,:,:]
                depth = input[:,3:,:,:]

            if args.modality == 'rgbd':
                rgb, depth, target, pred = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                # filename = output_directory + '/comparison_' + str(i) + '.png'
                filename_1 = output_directory + '/rgb/rgb_' + str(i) + '.png'
                filename_2 = output_directory + '/depth/depth_' + str(i) + '.png'
                filename_3 = output_directory + '/target/target_' + str(i) + '.png'
                filename_4 = output_directory + '/pred/pred_' + str(i) + '.png'
                utils.save_image(rgb, filename_1)
                utils.save_image(depth, filename_2)
                utils.save_image(target, filename_3)
                utils.save_image(pred, filename_4)
            else:
                img_merge = utils.merge_into_row(rgb, target, pred)

            # if i == 0:
            #     if args.modality == 'rgbd':
            #         img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred)
            #     else:
            #         img_merge = utils.merge_into_row(rgb, target, pred)
            # elif (i < 8*skip) and (i % skip == 0):
            #     if args.modality == 'rgbd':
            #         row = utils.merge_into_row_with_gt(rgb, depth, target, pred)
            #     else:
            #         row = utils.merge_into_row(rgb, target, pred)
            #     img_merge = utils.add_row(img_merge, row)
            # elif i == 8*skip:
            #     filename = output_directory + '/comparison_' + str(epoch) + '.png'
            #     utils.save_image(img_merge, filename)


        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge

if __name__ == '__main__':
    main()
