import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset, VideoDatasetV2
from network import C3D_model, R2Plus1D_model, R3D_model


import os

from gluoncv.torch.model_zoo import get_model

from gluoncv.torch.engine.config import get_cfg_defaults
# import torch


def load_model(model, cfg, load_fc=True):
    """
    Load pretrained model weights.
    """
    if os.path.isfile(cfg.CONFIG.MODEL.PRETRAINED_PATH):
        print("=> loading checkpoint '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))
        # if cfg.DDP_CONFIG.GPU is None:
        checkpoint = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH)
        # else:
        #     # Map model to be loaded to specified single gpu.
        #     loc = 'cuda:{}'.format(cfg.DDP_CONFIG.GPU)
        #     checkpoint = torch.load(cfg.CONFIG.MODEL.PRETRAINED_PATH, map_location=loc)
        model_dict = model.state_dict()
        if not load_fc:
        #     del model_dict['fc.weight']
        #     del model_dict['fc.bias']
            pretrained_dict = {k: v for k, v in checkpoint.items() if 'fc' not in k}
        else:
            pretrained_dict = checkpoint
        # unused_dict = {k: v for k, v in checkpoint.items() if not k in model_dict}
        # not_found_dict = {k: v for k, v in model_dict.items() if not k in checkpoint}
        # print("unused model layers:", unused_dict.keys())
        # print("pretrained_dict keys:", pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(cfg.CONFIG.MODEL.PRETRAINED_PATH, 'None'))
    else:
        print("=> no checkpoint found at '{}'".format(cfg.CONFIG.MODEL.PRETRAINED_PATH))

    return model, None


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.bn1, model.conv2, model.bn2, model.layer1, model.layer2, model.layer3]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.layer4, model.fc]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 100  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True # See evolution of the test set when training
nTestInterval = 20 # Run on test set every nTestInterval epochs
snapshot = 1 # Store a model every snapshot epochs
lr = 1e-4 # Learning rate

batch_size = 4
dataset = 'qpark_action_resz_rectV2_washframe' # Options: hmdb51 or ucf101

if dataset == 'hmdb51':
    num_classes=51
elif dataset == 'ucf101':
    num_classes = 2
elif 'qpark_action' in dataset:
    num_classes = 4
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'R2Plus1D' # Options: C3D or R2Plus1D or R3D
saveName = modelName+'-fintunelayer4_fc' + '-' + dataset+'_shift'

def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'R2Plus1D':
        # model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        cfg=get_cfg_defaults()
        cfg_file='./glucv_cfg_files/r2plus1d_v1_resnet50_kinetics400.yaml'
        cfg.merge_from_file(cfg_file)
        
        model = get_model(cfg)

        model, _ = load_model(model, cfg, load_fc=False)

        train_params = [
            {'params': get_1x_lr_params(model), 'lr': lr},
                        {'params': get_10x_lr_params(model), 'lr': lr * 10}]
    else:
        print('We only implemented C3D and R2Plus1D models.')
        raise NotImplementedError
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDatasetV2(dataset=dataset, split='train', clip_len=16, use_shift=True), batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDatasetV2(dataset=dataset, split='val',  clip_len=16), batch_size=batch_size, num_workers=4)
    test_dataloader  = DataLoader(VideoDatasetV2(dataset=dataset, split='test', clip_len=16), batch_size=batch_size, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)
    
    max_val_acc = 0
    tmp_val_acc = 0
    max_test_acc = 0
    tmp_test_acc = 0


    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
            else:
                tmp_val_acc = epoch_acc

                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1) and epoch>30:
                max_val_acc == tmp_val_acc

                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
                print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch > 30:
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)
            
            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


def test_model(epoch, test_dataloader, test_size):
    model_file_path='/DATA/disk1/phzhao/action/PyTorch_video_classifier/run/run_10/models/R2Plus1D-fintunelayer4_fc-qpark_action_resz_rectV2_washframe-flip_epoch-'+\
        str(epoch)+'.pth.tar'    
    cfg = get_cfg_defaults()
    cfg_file='./glucv_cfg_files/r2plus1d_v1_resnet50_kinetics400.yaml'
    cfg.merge_from_file(cfg_file)
        
    model = get_model(cfg)
    nEpochs=100

    model_dict = torch.load(model_file_path)['state_dict']

    model.load_state_dict(model_dict)
    criterion = nn.CrossEntropyLoss()
    # model.
    model.to(device)
    criterion.to(device)

    model.eval()

    start_time = timeit.default_timer()

    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects.double() / test_size

    # writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
    # writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)
    
    print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")


if __name__ == "__main__":
    state='train'
    # state='test'
    if state=='train':
        train_model()
    else:
        batch_size = 4
        dataset = 'qpark_action_resz_rectV2_washframe' # Options: hmdb51 or ucf101
        test_dataloader  = DataLoader(VideoDatasetV2(dataset=dataset, split='test', clip_len=16), batch_size=batch_size, num_workers=4)
        test_size = len(test_dataloader.dataset)
        

        for i in range(60,100):
            test_model(i, test_dataloader, test_size)
            test_model(i, test_dataloader, test_size)
            test_model(i, test_dataloader, test_size)

        # for i in range(10):
        #     test_model(45, test_dataloader, test_size)