import argparse
import gc
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.transform import Rotation
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Satellite
from model import PTR


def weights_init(self):
    if isinstance(self, nn.Linear):
        nn.init.kaiming_normal_(self.weight)


def test(args, net, test_loader, textio):

    net.eval()
    if args.test:
        model_path = "checkpoints/ptr/models/model.best.t7"
        net.load_state_dict(torch.load(model_path))

    num_examples = 0
    total_loss = 0
    time_costs = 0
    rotations_ab = []
    rotations_ab_pred = []
    translations_ab = []
    translations_ab_pred = []

    for src, target, rotation_ab, translation_ab in tqdm(test_loader):
        src = src.cuda()
        target = target.cuda()

        batch_size = src.size(0)
        num_examples += batch_size

        rotation_ab_pred, translation_ab_pred, time_cost = net(src, target)

        translation_ab_pred = translation_ab_pred.squeeze(1)
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())

        identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab.cuda()), identity) \
            + loss_fn(translation_ab_pred, translation_ab.cuda())

        total_loss += loss.item() * batch_size
        time_costs = time_costs + time_cost

    time_costs = time_costs / num_examples
    test_loss = total_loss * 1. / num_examples
    rotations_ab = np.asarray(rotations_ab).reshape(num_examples, 3, 3)
    rotations_ab_pred = np.asarray(rotations_ab_pred).reshape(num_examples, 3, 3)
    translations_ab = np.asarray(translations_ab).reshape(num_examples, 3)
    translations_ab_pred = np.asarray(translations_ab_pred).reshape(num_examples, 3)
    angles_error = np.rad2deg(np.arccos((np.trace(np.matmul(rotations_ab, rotations_ab_pred.transpose(0, 2, 1)), axis1=1, axis2=2) - 1) / 2))
    for i in range(int(np.shape(angles_error)[0])):
        if math.isnan(angles_error[i]):
            angles_error[i] = 90
    success_ratio_1 = (angles_error < 1).sum() / num_examples
    success_ratio_5 = (angles_error < 5).sum() / num_examples
    success_ratio_10 = (angles_error < 10).sum() / num_examples
    success_ratio_15 = (angles_error < 15).sum() / num_examples
    success_ratio_20 = (angles_error < 20).sum() / num_examples
    success_ratio_40 = (angles_error < 40).sum() / num_examples
    success_ratio_60 = (angles_error < 60).sum() / num_examples
    angles_error = angles_error.mean()
    translation_error = np.linalg.norm(translations_ab - translations_ab_pred, axis=1)
    for i in range(int(np.shape(translation_error)[0])):
        if translation_error[i] > 1:
            translation_error[i] = 1
    translation_error = translation_error.mean()

    textio.write('==Final Test==' + '\n')
    textio.write('A-->B' + '\n')
    textio.write('Loss: %f, Angle_Error : %f, T_Error : %f, success_ratio_1 : %f, success_ratio_5 : %f, '
                 'success_ratio_10 : %f, success_ratio_15 : %f, success_ratio_20 : %f, success_ratio_40 : %f, '
                 'success_ratio_60 : %f, time_cost : %f' %
                 (test_loss, angles_error, translation_error, success_ratio_1, success_ratio_5, success_ratio_10,
                  success_ratio_15, success_ratio_20, success_ratio_40, success_ratio_60, time_costs) + '\n')
    textio.flush()

    return test_loss, angles_error, translation_error


def train_one_epoch(args, net, train_loader, opt):

    net.train()
    if args.model_path:
        net.load_state_dict(torch.load(args.model_path))

    num_examples = 0
    total_loss = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []
    angles_ab = []
    angles_ab_pred = []

    for src, target, rotation_ab, translation_ab in tqdm(train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size

        rotation_ab_pred, translation_ab_pred, time_cost = net(src, target)

        translation_ab_pred = translation_ab_pred.squeeze(1)
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        angle_ab = Rotation.from_matrix(rotation_ab.detach().cpu().numpy())
        angle_ab = angle_ab.as_euler('ZYX', degrees=True)
        angles_ab.append(angle_ab)
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        angle_ab_pred = Rotation.f
        angle_ab_pred = angle_ab_pred.as_euler('ZYX', degrees=True)
        angles_ab_pred.append(angle_ab_pred)
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())

        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) + \
            loss_fn(translation_ab_pred, translation_ab)
        loss.backward()
        opt.step()  # apply gradients
        total_loss += loss.item() * batch_size

    train_loss = total_loss * 1. / num_examples
    rotations_ab = np.asarray(rotations_ab).reshape(num_examples, 3, 3)
    rotations_ab_pred = np.asarray(rotations_ab_pred).reshape(num_examples, 3, 3)
    translations_ab = np.asarray(translations_ab).reshape(num_examples, 3)
    translations_ab_pred = np.asarray(translations_ab_pred).reshape(num_examples, 3)
    angles_error = np.arccos((np.trace(np.matmul(rotations_ab, rotations_ab_pred.transpose(0, 2, 1)), axis1=1, axis2=2) - 1) / 2)
    angles_error = np.rad2deg(angles_error).mean()
    translation_error = np.linalg.norm(translations_ab - translations_ab_pred, axis=1).mean()

    return train_loss, angles_error, translation_error


def train(args, net, train_loader, test_loader, textio, boardio):

    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(opt, milestones=[40, 80], gamma=0.5)
    net.apply(weights_init)

    best_test_loss = np.inf

    for epoch in range(args.epochs):

        train_loss, angles_error_train, translation_error_train = train_one_epoch(args, net, train_loader, opt)
        gc.collect()
        with torch.no_grad():
            test_loss, angles_error_test, translation_error_test = test(args, net, test_loader, textio)

        if best_test_loss >= test_loss:

            best_test_loss = test_loss

            torch.save(net.state_dict(), 'checkpoints/ptr/models/model.best.t7')

        textio.write('==Train==' + '\n')
        textio.write('A-->B' + '\n')
        textio.write('EPOCH:: %d, Loss: %f, Angle_Error : %f, T_Error : %f' % (epoch, train_loss, angles_error_train,
                                                                               translation_error_train) + '\n')
        textio.write('==Test==' + '\n')
        textio.write('A-->B' + '\n')
        textio.write('Loss: %f, Angle_Error : %f, T_Error : %f' % (test_loss, angles_error_test, translation_error_test)
                                                               + '\n')
        textio.flush()

        boardio.add_scalar('A-B/train/loss', train_loss, epoch)
        boardio.add_scalar('A-B/train/Angle_Error', angles_error_train, epoch)
        boardio.add_scalar('A-B/train/T_Error', translation_error_train, epoch)
        boardio.add_scalar('A-B/test/loss', test_loss, epoch)
        boardio.add_scalar('A-B/test/Angle_Error', angles_error_test, epoch)
        boardio.add_scalar('A-B/test/T_Error', translation_error_test, epoch)
        boardio.add_scalar('A-B/best_test/loss', best_test_loss, epoch)

        if epoch % 10 == 0:
            torch.save(net.state_dict(), 'checkpoints/ptr/models/model.%d.t7' % epoch)
        scheduler.step()
        gc.collect()  # clear memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--epochs', type=int, default=121)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--emb_dims', type=int, default=64)
    parser.add_argument('--num_points_src', type=int, default=1024)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--ff_dims', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    parser.add_argument('--test', action='store_true', default=False, help='evaluate the model')
    parser.add_argument('--icp', action='store_true', default=False, help='apply icp algorithm')
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True

    boardio = SummaryWriter(logdir='checkpoints/ptr')
    textio = open('checkpoints/ptr/run.log', 'a')
    textio.write(str(args) + '\n')
    textio.flush()

    train_loader = DataLoader(Satellite(num_points_src=args.num_points_src, partition='train', gaussian_noise=False),
                              batch_size=args.batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(Satellite(num_points_src=args.num_points_src, partition='eval', gaussian_noise=False),
                             batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(Satellite(num_points_src=args.num_points_src, partition='test', gaussian_noise=False),
                             batch_size=args.batch_size, shuffle=False, drop_last=False)

    net = PTR(args)
    net = net.cuda()
    print("Initiate")
    if args.test:
        test(args, net, test_loader, textio)
    else:
        train(args, net, train_loader, eval_loader, textio, boardio)
    print('Finish')
    textio.close()
    boardio.close()


