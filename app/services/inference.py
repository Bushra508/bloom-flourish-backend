import torch
from app.core.model import load_model, device

def predict(image_tensor):
    model, class_names, pretty_names = load_model()

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(probs, 3)

    results = []
    for i in range(3):
        idx = top_indices[0][i].item()

        results.append({
            "class_id": class_names[idx],
            "label": pretty_names[idx],
            "confidence": float(top_probs[0][i].item())
        })
    
    return results