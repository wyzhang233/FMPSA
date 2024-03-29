import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import random
from dataset import HyperDatasetValid, HyperDatasetTrain1, HyperDatasetTrain2, HyperDatasetTrain3, \
    HyperDatasetTrain4  # Clean Data set
from FMPSA import FMPSA
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, LossTrainCSS, Loss_valid
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True
start = time.perf_counter()

parser = argparse.ArgumentParser(description="SSR")
parser.add_argument("--batchSize", type=int, default=4, help="batch size")
parser.add_argument("--end_epoch", type=int, default=40+1, help="number of epochs")
parser.add_argument("--init_lr", type=float, default=1e-4, help="initial learning rate")
parser.add_argument("--decay_power", type=float, default=1.5, help="decay power")
parser.add_argument("--trade_off", type=float, default=10, help="trade_off")
parser.add_argument("--max_iter", type=float, default=300000, help="max_iter")
parser.add_argument("--outf", type=str, default="CleanResults", help='path log files')
opt = parser.parse_args()


def main(): #定义主函数
    cudnn.benchmark = True

    print("\nloading dataset ...")
    train_data1 = HyperDatasetTrain1(mode='train')
    train_data2 = HyperDatasetTrain2(mode='train')
    train_data3 = HyperDatasetTrain3(mode='train')
    train_data4 = HyperDatasetTrain4(mode='train')
    print("Train1:%d,Train2:%d,Train3:%d,Train4:%d," % (len(train_data1), len(train_data2), len(train_data3), len(train_data4),))
    val_data = HyperDatasetValid(mode='valid')
    print("Validation set samples: ", len(val_data))

    train_loader1 = DataLoader(dataset=train_data1, batch_size=opt.batchSize, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    train_loader2 = DataLoader(dataset=train_data2, batch_size=opt.batchSize, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    train_loader3 = DataLoader(dataset=train_data3, batch_size=opt.batchSize, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    train_loader4 = DataLoader(dataset=train_data4, batch_size=opt.batchSize, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    train_loader = [train_loader1, train_loader2, train_loader3, train_loader4]
    val_loader = DataLoader(dataset=val_data, batch_size=1,  shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    print("\nbuilding models_baseline ...")
    model = FMPSA(3, 31, 128, 8)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    criterion_train = LossTrainCSS()
    criterion_valid = Loss_valid()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()
        criterion_train.cuda()
        criterion_valid.cuda()

    start_epoch = 0
    iteration = 0
    record_val_loss = 1000
    optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)
    loss_csv = open(os.path.join(opt.outf, 'loss.csv'), 'a+')
    log_dir = os.path.join(opt.outf, 'train.log')
    logger = initialize_logger(log_dir)

    resume_file = ''
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            iteration = checkpoint['iter']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(start_epoch+1, opt.end_epoch):
        start_time = time.time()
        train_loss, iteration, lr = train(train_loader, model, criterion_train, optimizer, epoch, iteration, opt.init_lr, opt.decay_power, opt.trade_off)
        val_loss = validate(val_loader, model, criterion_valid)

        if torch.abs(val_loss - record_val_loss) < 0.0001 or val_loss < record_val_loss:
            save_checkpoint(opt.outf, epoch, iteration, model, optimizer)
            if val_loss < record_val_loss:
                record_val_loss = val_loss

        end_time = time.time()
        epoch_time = end_time - start_time
        print("Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f "
              % (epoch, iteration, epoch_time, lr, train_loss, val_loss))
        record_loss(loss_csv,epoch, iteration, epoch_time, lr, train_loss, val_loss)
        logger.info("Epoch [%02d], Iter[%06d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f "
                    % (epoch, iteration, epoch_time, lr, train_loss, val_loss)) #写入日志信息

def train(train_loader, model, criterion, optimizer, epoch, iteration, init_lr, decay_power, trade_off):
    model.train()
    random.shuffle(train_loader)
    losses = AverageMeter()
    losses_rgb = AverageMeter()
    for k, train_data_loader in enumerate(train_loader):
        for i, (images, labels) in enumerate(train_data_loader):
            labels = labels.cuda()
            images = images.cuda()
            images = Variable(images)
            labels = Variable(labels)
            lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=opt.max_iter, power=decay_power)
            iteration = iteration + 1

            output = model(images)
            loss, loss_rgb = criterion(output, labels, images)
            loss_all = loss + trade_off * loss_rgb
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            losses.update(loss.data)
            losses_rgb.update(loss_rgb.data)
            print('[Epoch:%02d],[Process:%d/%d],[iter:%d],lr=%.9f,train_losses.avg=%.9f, rgb_train_losses.avg=%.9f'
                  % (epoch, k+1, len(train_loader), iteration, lr, losses.avg, losses_rgb.avg))

    return losses.avg, iteration, lr

def validate(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
        losses.update(loss.data)

    return losses.avg

def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1, max_iter=100, power=0.9):

    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr*(1 - iteraion/max_iter)**power
    for param_group in optimizer.param_groups: #参数
        param_group['lr'] = lr

    return lr

if __name__ == '__main__':
    main()
    print(torch.__version__)
    end = time.perf_counter()
    print("运行耗时", end-start)