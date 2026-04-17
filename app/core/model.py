import torch
from efficientnet_pytorch import EfficientNet

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# store metadata globally
class_names = None
pretty_names = None

def load_model():
    global model, class_names, pretty_names

    if model is None:
        checkpoint = torch.load("model/plant_model.pth", map_location=device)

        num_classes = checkpoint['num_classes']

        # 🔥 recreate model EXACTLY like training
        model = EfficientNet.from_name('efficientnet-b0')
        model._fc = torch.nn.Linear(model._fc.in_features, num_classes)

        # load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)
        model.eval()

        # store labels
        class_names = checkpoint['class_names']
        pretty_names = checkpoint['pretty_names']

    return model, class_names, pretty_names