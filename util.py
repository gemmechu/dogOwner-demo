import torch
import torch.nn.functional as F
from torchvision import transforms
import config
from boot import app


class ModelLoaded:
    model = None
    acc = 0

    @staticmethod
    def get_model():
        if not ModelLoaded.model:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            selected_model = app.config['USE_MODEL']
            model = selected_model()
            model.to(device)
            state = torch.load(config.path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state["state_dict"])
        return model


trfrm = transforms.Compose([
    lambda x: x.convert('RGB'),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
topil = transforms.ToPILImage()
totensor = transforms.Compose(trfrm.transforms[:-1])


def get_distance(img1, img2):
    model = ModelLoaded.get_model()
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        x1 = trfrm(img1).unsqueeze(0)
        x2 = trfrm(img2).unsqueeze(0)
        x1,x2 = x1.to(device), x2.to(device)
        embed1 = model(x1)
        embed2 = model(x2)
        return F.pairwise_distance(embed1, embed2)


def is_same(img1, img2, threshold=0.5):
    distance = get_distance(img1, img2)
    return distance, distance <= threshold
