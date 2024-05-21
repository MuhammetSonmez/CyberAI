import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from CyberAI import ChatDataset, Cyber, nlpp
torch.manual_seed(42)

#HyperParameters

batch_size = 8

hidden_size = 8
output_size = len(nlpp.tags)
input_size = len(nlpp.X_train[0])

learning_rate = 0.001

num_epochs = 850


print(input_size, len(nlpp.all_words))
print(output_size, nlpp.tags)
dataset = ChatDataset(len(nlpp.X_train), nlpp.X_train, nlpp.y_train)

train_loader = DataLoader(dataset=dataset, batch_size = batch_size, shuffle=True) # , num_workers = 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Cyber(input_size, hidden_size, output_size)

# loss and optimizer
ciriterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# train loop
for epoch in range(num_epochs):

  for (words, labels) in train_loader:
    words = words.to(device)
    labels = labels.to(device)

    #forward
    outputs = model(words)
    loss = ciriterion(outputs, labels)

    #backward and optimizer
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

  if (epoch +1) %50 == 0:
    print(f"epcoh {epoch+1}/{num_epochs}, loss = {loss.item():.4f}")
print(f'final loss, loss = {loss.item():.4f}')

data = {
    "model_state":model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": nlpp.all_words,
    "tags": nlpp.tags
}

FILE = r"models\cyberAI.pth"

torch.save(data, FILE)

print(f'Model saved into -> {FILE}')

