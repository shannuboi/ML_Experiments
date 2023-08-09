import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from ResNetModel import ResBlock, ResNet18    

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vegitables = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli',
                  'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber',
                  'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']
    

    image_path = 'tomato2.jpg' # <-- File name here

    veg_image = Image.open(image_path)
    image_size = 90
    mean, std = torch.Tensor([0.4689, 0.4635, 0.3433]), torch.Tensor([0.2327, 0.2274, 0.2420])
    transformer = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((image_size, image_size), antialias=None),
        torchvision.transforms.Normalize(mean, std)
    ])
    input_image = transformer(veg_image)

    checkpoint = torch.load("best_val_checkpoint.pth")
    model = ResNet18(3, ResBlock, 15).to(device)
    model.load_state_dict(checkpoint['model'])

    prediction = model(torch.unsqueeze(input_image, 0).to(device))
    prediction = torch.argmax(prediction, 1)
    prediction = vegitables[prediction]
    
    plt.imshow(veg_image)   
    plt.title('Prediction: ' + prediction) 
    plt.axis('off')
    plt.show()
    veg_image.close()


if __name__ == '__main__':
    main()