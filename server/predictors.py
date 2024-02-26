import torch


def predict_digit(input, model_path):
    model = torch.load(model_path)
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(input, dtype = torch.float32))
    return output.argmax().item()