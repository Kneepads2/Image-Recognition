from flask import Flask, request, render_template
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

app = Flask(__name__)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)  #100 classes for CIFAR100

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

net = Net()

trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"[{epoch + 1}] loss: {running_loss:.3f}")

torch.save(net.state_dict(), "cifar100_net.pth")

net.load_state_dict(torch.load("cifar100_net.pth", map_location=torch.device('cpu')))
net.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True)
class_names = cifar100_train.classes

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        label = request.form['label'].strip().lower()
        image_file = request.files['image']
        if not image_file:
            return "No image uploaded", 400

        image = Image.open(image_file).convert('RGB')
        image = transform(image).unsqueeze(0)


        with torch.no_grad():
            outputs = net(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
            is_match = (predicted_class.lower() == label)

        prediction = {
            "requested": label,
            "predicted": predicted_class,
            "match": "Yes" if is_match else "No",
            "image":image_file
        }

    return render_template('index.html', prediction=prediction)

# --- Run ---
if __name__ == '__main__':
    app.run(debug=True)
