import utils
import torch
from torch.autograd import Variable


class Prediction:
    def __init__(self, index, raw_pred, pred, correct, target):
        self.index = index
        self.raw_pred = raw_pred
        self.pred = pred
        self.correct = correct
        self.target = target

    def __str__(self):
        return str((self.index, self.pred, self.target))

    def __repr__(self):
        return str((self.index, self.pred, self.target))


def run_net_batch(net, opt, dataset, converter):
    print('Starting new batch result')

    image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
    if opt.cuda:
        image = image.cuda()
    image = Variable(image)

    text = torch.IntTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)
    text = Variable(text)
    length = Variable(length)

    # Play with the batch size for the time optimization
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    n_correct = 0
    predicted_list = []

    max_iter = len(data_loader)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1

        cpu_images, cpu_texts, data_indexes = data
        print(data_indexes)

        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        _, preds = preds.max(2)

        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        raw_preds = converter.decode(preds.data, preds_size.data, raw=True)

        for pred, raw_pred, target, data_point_index in zip(sim_preds, raw_preds, cpu_texts, data_indexes):

            predicted_list.append(Prediction(data_point_index, raw_pred, pred, pred == target.lower(), target))

            if pred.lower() == target.lower():
                n_correct += 1

    accuracy = n_correct / float(len(dataset))
    print('Accuray: %f' % (accuracy))

    return accuracy, predicted_list


def val_batch(net, opt, dataset, converter, criterion, max_iter=100):
    print('Starting new val batch')

    image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
    if opt.cuda:
        image = image.cuda()
    image = Variable(image)

    text = torch.IntTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)
    text = Variable(text)
    length = Variable(length)

    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

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

        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size  # Todo how is the cost calculated?
        loss_avg.add(cost)

        _, preds = preds.max(2)  # todo where is the output size set to 26? Empirically it is.
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

    return loss_avg.val(), accuracy


def train_batch(net, criterion, optimizer, train_iter, opt, converter):

    image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
    if opt.cuda:
        image = image.cuda()
    image = Variable(image)

    text = torch.IntTensor(opt.batchSize * 5)
    length = torch.IntTensor(opt.batchSize)
    text = Variable(text)
    length = Variable(length)

    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts) #Todo is this conversion correct?

    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = net(image) #todo average time of run?
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size #TODO what is this criterion
    net.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


