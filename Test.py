import torch
import numpy as np
import pandas as pd
from LoadData import TestDataSet
from torch.utils.data import DataLoader

model = torch.load("trained_model.pth")
test_dataset = TestDataSet("dataset/test.csv")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
submission = []
i = 1
for data in test_dataloader:
    output = model(data)
    category = output.argmax(1)
    submission.append([i, category.item()])
    i += 1
submission = pd.DataFrame(submission)
submission.columns = ['ImageId', 'Label']
submission.to_csv('submission.csv')
