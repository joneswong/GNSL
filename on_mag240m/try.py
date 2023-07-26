from pytorch_lightning.metrics import Metric
from torchmetrics import Accuracy
train_acc = Accuracy()
#train_acc(, torch.Tensor([1, 1, 0]))
print(isinstance(train_acc, Metric))
print(isinstance(train_acc, list))
