import os
import argparse
import numpy as np
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM
import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from efficientnet_pytorch import EfficientNet

from utils import *
from dataset import PathDataset


batch_size = 8
lr = 0.01
num_epochs = 8
num_classes = 2
seed = '2020'
model_name = 'efficientnet-b7'
momentum = 0.9
weight_decay = 5e-04
stepsize = 6
is_scheduler = True
gamma = 0.5
print_freq = 100

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(),os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
        model.eval()
        print('model loaded!')

    def infer(image_path):
        result = []
        with torch.no_grad():             
            batch_loader = DataLoader(dataset=PathDataset(image_path, labels=None),
                                        batch_size=batch_size,shuffle=False)

            # Train the model 
            for i, images in enumerate(batch_loader):
                y_hat = model(images.cuda()).cpu().numpy()
                y_hat[:,0] += 0.2
                result.extend(np.argmax(y_hat, axis=1))

        print('predicted')
        return np.array(result)

    nsml.bind(save=save, load=load, infer=infer)

def train(model, criterion, optimizer, batch_loader, epoch):
    model.train()
    losses = AverageMeter() # loss의 평균을 구하는 함수
    for i, (images, labels) in enumerate(batch_loader):
        images = images.cuda()
        labels = labels.cuda()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), labels.size(0)) # labels.size(0) = batch size

        if (i+1) % print_freq == 0: #매 print_freq iteration마다 아래를 출력
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})\t lr {}" \
                    .format(i+1, len(batch_loader), losses.val, losses.avg, lr))

        if (i+1) % 501 == 0 :
            if epoch > 4 :
                name = str(epoch) + '_' + str(i+1)
                nsml.save(name)
                print('save :', name)

    nsml.report(summary=True, step=epoch, epoch_total=num_epochs, loss=loss.item())#, acc=train_acc)
    nsml.save(epoch)

def test(model, testloader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad(): 
        for i, (data, labels) in enumerate(testloader):
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()
            # if i == 100:
            #     break
    acc = correct.item() * 1.0 / total
    err = 1.0 - acc
    return acc, err


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()

    torch.manual_seed(seed)
    use_gpu = torch.cuda.is_available()
    print("Use gpu :", use_gpu)
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)

    print("Creating model: {}".format(model_name))
    model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    print("End Creating model: {}".format(model_name))

    print("Loading model to GPU")
    # model = nn.DataParallel(model).to("cuda")
    model.to("cuda")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    if is_scheduler == True:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

    bind_model(model)
    if config.pause: ## test mode 일때는 여기만 접근
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if config.mode == 'train': ### training mode 일때는 여기만 접근
        print('Training Start...')

        root_path = os.path.join(DATASET_PATH,'train')
        image_keys, image_path = path_loader(root_path)
        labels = label_loader(root_path, image_keys)
        batch_loader = DataLoader(dataset=PathDataset(image_path, labels, test_mode=False), batch_size=batch_size, shuffle=True)

        # Train the model
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs-1))
            #model train
            train(model, criterion, optimizer, batch_loader, epoch)

            if is_scheduler == True: 
                scheduler.step()

            if epoch >= 6 :
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.0001

            if epoch > 5:
                acc, err = test(model, batch_loader)
                print("Accuracy (%): {}\t Error rate(%): {}".format(acc, err))