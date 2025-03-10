from .Resnet import resnet18, resnet34

model_dict = {
    "resnet18": resnet18,
    "resnet34": resnet34
}

def create_model(model_name, num_classes):
    return model_dict[model_name](num_classes)
