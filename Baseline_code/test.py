from loss import *
from sklearn.metrics import accuracy_score
loss_fn = CustomLoss()
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    loss_fn = CustomLoss()
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    loss = loss_fn(labels, preds)
    return {
        'accuracy': round(acc, 4),
        'loss':round(loss,4),
    }
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)

print(accuracy_score(input, target))