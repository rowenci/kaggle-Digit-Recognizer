import torch
from torch.utils.tensorboard import SummaryWriter

from LoadData import TrainDataSet
from torch.utils.data import DataLoader
from BasicCNN import BasicCNN
import torch.optim as optim
import torch.nn as nn


# super parameters
batch_size = 32
lr = 0.03
epoches = 10

train_dataset = TrainDataSet("datas/train.csv")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)



model = BasicCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)


writer = SummaryWriter('logs')

total_train_step = 0
total_test_step = 0
for epoch in range(epoches):
    model.train()
    print('epoch {} begin'.format(epoch + 1))
    for data in train_dataloader:
        x, label = data
        y = model(x)
        print(label)
        print(label.shape)
        print(y.shape)
        loss = criterion(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print('training number {}, loss : {}'.format(total_train_step, loss.item()))
        writer.add_scalar('train_loss', loss.item(), total_train_step)

torch.save(model, 'trained_model.pth')
writer.close()
