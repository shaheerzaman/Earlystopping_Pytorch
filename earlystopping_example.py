import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def create_datasets(batch_size):
    #percentage of traning set to use as validation
    valid_size = 0.2

    transform = transforms.ToTensor()

    train_data = datasets.MNIST(root='data', 
                                train=True, 
                                download=True, 
                                transform=transform)

    test_data = datasets.MNIST(root='data', 
                                train=False, 
                                download=True, 
                                transform=transform)
    
    num_train = len(train_data)
    indices = list(range(num_range))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    #load the training data in batches
    train_loader = torch.utils.data.DataLoader(train_data, 
                                              batch_size=batch_size,
                                              sampler=train_sampler, 
                                              num_workers=0)

    valid_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=batch_size,
                                            sampler=valid_sampler,
                                            num_workers=0)

    test_loader = torch.utils.data.DataLoader(test_data, 
                                            batch_size=batch_size, 
                                            num_workers=0)

    return train_loader, test_loader, valid_loader


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return  x

model = Net()
print(model)

#loss 
criterion = nn.CrossEntropyLoss()

#optimizer
optimizer = torch.optim.Adam(model.parameters())

from pytool import EarlyStopping

def train_model(model, batch_size, patience, n_epochs):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs+1):
        model.train()
        for batch, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        for data, target in valid_loader:
            output = model(data)
            loss = criterion(output, target)
            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print('Early stopping')
            break

    model.load_state_dict(torch.load('checkpoint.pt'))
    
    return model, avg_train_losses, avg_valid_losses
    
batch_size = 256
n_epochs = 100

train_loader, test_loader, valid_loader = create_datasets(batch_size)

patience = 20

model, train_loss, valid_loss = train_model(model, batch_size, patience, 
                                n_epochs)

#visualize the loss as the network trained
fig = plt.figure(figsize=(10, 8))
plt.plot(range(1, len(train_loss)+1), train_loss, label='Training loss')
plt.plot(range(1, len(valid_loss)+1), valid_loss, label='Valiation loss')

#find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss)) + 1
plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 0.5) #consistent scale
plt.xlim(0, len(train_loss)+1) #consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('loss_plot.png', bbox_inces='tigh')

#Test the trained network
test_loss = 0.0
class_correct = list(0 for i in range(10))
class_total = list(0 for i in range(10))

model.eval()
for data, target in test_loader:
    if len(target.data) != batch_size:
        break

for data, target in test_loader:
    if len(target.data) != batch_size::
        break
    
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

#calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}'.format(test_loss))

for i in range(10):
    if class_total[i]>0:
        print('Test Accuracy of %5s: %2d%% (%2d%2d)' %(
            str(i), 100*class_correct[i] / class_total[i], 
            class_correct[i], class_total[i]
        ))
    else:
        print('Test Accuracy of %5s: N/A (no training exams)' % (classes[i]))
        
print('\n Test Accuracy (overall): %2d%% (%2d/%2d)', %(
    100*np.sum(class_correct)/np.sum(class_total), 
    np.sum(class_correct), np.sum(class_total)
))

#visualize sample Test Results
dataiter = iter(test_loader)
images, labels = dataiter.next()

#get the sample outputs
output = model(images)
_, preds = torch.max(output, 1)
images = images.numpy()

fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title('{} ({})'.format(str(preds[idx].item()), str(labels[idx].item())),
    color=('green' if preds[idx]==labels[idx] else 'red'))





            






                                                                                                                             
    

                                