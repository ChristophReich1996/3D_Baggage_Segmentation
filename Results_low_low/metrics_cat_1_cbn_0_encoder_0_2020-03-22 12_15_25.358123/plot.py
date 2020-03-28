import matplotlib.pyplot as plt
import matplotlib2tikz
import torch
import numpy as np

bb_iou = torch.load('validation_bb_iou.pt').numpy()

plt.plot(bb_iou)
plt.grid()
plt.xlabel('Training Epochs')
plt.ylabel('BB Iou')
matplotlib2tikz.save('bb_iou_cat_no_cbn.tex', figureheight = '\\figH', figurewidth = '\\figW')
plt.show()