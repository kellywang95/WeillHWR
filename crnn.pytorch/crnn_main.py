from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset

from models import crnn as crnn_model

from model_code import train_batch


# TODO how does it go from a-----v--a-i-l-a-bb-l-ee-- => available

# TODO - Note from the repo. Construct dataset following origin guide. For training with variable length, please sort the image according to the text length.
# first point - How do we do it now?
# easy change to other thing?


def main(opt):
    print(opt)

    if opt.experiment is None:
        opt.experiment = 'expr'

    os.system('mkdir {0}'.format(opt.experiment))

    # TODO why is this?
    opt.manualSeed = random.randint(1, 10000)  # fix seed

    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    train_dataset = dataset.hwrDataset(mode="train")
    assert train_dataset
    # if not opt.random_sample:
    #     sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
    # else:
    #     sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=True))
    # test_dataset = dataset.lmdbDataset(
    #     root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))

    test_dataset = dataset.hwrDataset(mode="test", transform=dataset.resizeNormalize((100, 32)))

    nclass = len(opt.alphabet) + 1
    nc = 1

    criterion = CTCLoss()

    # custom weights initialization called on crnn
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    crnn = crnn_model.CRNN(opt.imgH, nc, nclass, opt.nh)
    crnn.apply(weights_init)
    if opt.crnn != '':
        print('loading pretrained model from %s' % opt.crnn)
        crnn.load_state_dict(torch.load(opt.crnn))
    print(crnn)

    # TODO make this central

    image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
    text = torch.IntTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)

    if opt.cuda:
        crnn.cuda()
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
        image = image.cuda()
        criterion = criterion.cuda()

    image = Variable(image)
    text = Variable(text)
    length = Variable(length)

    # TODO what is this, read this.
    # loss averager
    loss_avg = utils.averager()

    # Todo default is RMS Prop. I wonder why?
    # setup optimizer

    #Following the paper's recommendation

    opt.adadelta = True
    if opt.adam:
        optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                               betas=(opt.beta1, 0.999))
    elif opt.adadelta:
        optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
    else:
        optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

    converter = utils.strLabelConverter(opt.alphabet)

    def val(net, dataset, criterion, max_iter=100):
        print('Start val')

        for p in crnn.parameters():
            p.requires_grad = False

        net.eval()
        data_loader = torch.utils.data.DataLoader(
            dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
        val_iter = iter(data_loader)

        i = 0
        n_correct = 0
        loss_avg = utils.averager()

        max_iter = min(max_iter, len(data_loader))
        for i in range(max_iter):
            data = val_iter.next()
            i += 1
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            t, l = converter.encode(cpu_texts)
            utils.loadData(text, t)
            utils.loadData(length, l)

            preds = crnn(image)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            cost = criterion(preds, text, preds_size, length) / batch_size
            loss_avg.add(cost)

            _, preds = preds.max(2)
            # preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, cpu_texts):
                if pred == target.lower():
                    n_correct += 1

        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
            print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

        accuracy = n_correct / float(max_iter * opt.batchSize)
        print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


    for epoch in range(opt.niter):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            for p in crnn.parameters():
                p.requires_grad = True
            crnn.train()

            cost = train_batch(crnn, criterion, optimizer, train_iter, opt, converter)
            loss_avg.add(cost)
            i += 1

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            if i % opt.valInterval == 0:
                try:
                    val(crnn, test_dataset, criterion)
                except Exception as e:
                    print(e)

            # do checkpointing
            if i % opt.saveInterval == 0:
                torch.save(
                    crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainroot', help='path to dataset')
    parser.add_argument('--valroot', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input vto network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--crnn', default='data/crnn.pth', help="path to crnn (to continue training)")
    parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
    parser.add_argument('--valInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--saveInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--random_sample', action='store_true',
                        help='whether to sample the dataset with random sampler')
    opt = parser.parse_args()

    main(opt)