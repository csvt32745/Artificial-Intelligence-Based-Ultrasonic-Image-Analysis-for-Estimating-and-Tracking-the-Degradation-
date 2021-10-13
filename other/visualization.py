import io
from typing import Optional
import numpy as np
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from train_src.dataloader import denormalize_img_rgb

def Vis_Input_Merge_Heat(input: np.ndarray, mask: np.ndarray, GT: np.ndarray, heatmap: np.ndarray, name: Optional[str] = ""):
    '''
        input: (h, w, 3), [0, 1]
        mask: (h, w), {0, 1}
        GT: (h, w), {0, 1}
        heatmap: (h, w), [0, 1]
        name(opt): str
        return: (h', w' 3) img of visualization
    '''
    input = input
    fontsize = 16
    dpi = 80
    fig, axs = plt.subplots(1, 3, figsize=(430*3/dpi, 400/dpi), gridspec_kw={'width_ratios': [1, 1, 1]}, dpi=dpi)
    for ax in axs:
        ax.set_axis_off()
    # 0: Input image
    input = denormalize_img_rgb(input)
    axs[0].set_title('Input', fontsize=fontsize)
    axs[0].imshow(input)
    
    # 1: Merge of labels
    axs[1].set_title('Result', fontsize=fontsize)
    colors = [
        mpatches.Patch(color=(1, 0, 0), label='Result'),
        mpatches.Patch(color=(0, 0, 1), label='GT'),
    ]
    labels = input.copy()
    labels[..., 0] += mask*0.5 # red
    labels[..., 2] += GT*0.5 # blue

    axs[1].legend(handles = colors, loc='lower right', fontsize=fontsize)
    axs[1].imshow(np.clip(labels, 0, 1))
    
    # 2: Heatmap of prob
    axs[2].set_title('Heatmap', fontsize=fontsize)
    hmap = sns.heatmap(heatmap, alpha = 0.3, zorder=2)
    hmap.imshow(input,
          aspect = hmap.get_aspect(),
          extent = hmap.get_xlim() + hmap.get_ylim(),
          zorder = 1)

    fig.suptitle(str(name), fontsize=14, va='top', y=0.06)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    with io.BytesIO() as buf:
        fig.savefig(buf, format='raw')
        buf.seek(0)
        data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    data = data.reshape(int(h), int(w), -1)
    plt.close()
    return data