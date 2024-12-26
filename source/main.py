import warnings

from torch import device, cuda
from MicroService import AnnotationClassifier, BookRegressor, Cleaner, MicroService

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    DEVICE = torch.device("cpu")
    print(DEVICE)

    ac = AnnotationClassifier(
        model_name=TEXTMODELCLASSIFIEROPTION['model'],
        n_classes=TEXTMODELCLASSIFIEROPTION['n_classes'],
        max_len=TEXTMODELCLASSIFIEROPTION['max_len'],
        device=DEVICE
    ).load('models/LaBSE_CrossEntropyLoss_two.pt')

    br = BookRegressor(input_dim=171, DEVICE).load('models/best_model_0_758.pt')

    cleaner = Cleaner()

    ms = MicroService(cleaner, ac, br, DEVICE)

