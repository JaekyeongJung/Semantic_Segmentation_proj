import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()
    

def decode_segmap(pred):
    label_colors = np.array([10, 20, 30, 40, 50, 60, 70, 80, 100])
    pred = np.argmax(pred, axis=0)
    RGB = np.zeros((3, pred.shape[0], pred.shape[1]))
    
    for label in range(9):
        for c in range(3):
            RGB[c] = np.where(pred == label, label_colors[label], RGB[c])
    segmap = RGB.astype(np.uint8)
    # segmap = np.array(segmap * 255).astype(np.uint8)
    
    # print(segmap)
    # segmap = np.transpose(segmap, (1,2,0))
    # pil_image=Image.fromarray(segmap)
    # pil_image.show()
    
    return segmap

if __name__=='__main__':
    pred = (np.random.rand(9,256,256)*10).astype(np.uint8)
    segmap = decode_segmap(pred)
    print("segmap shape : ", segmap[0])
    
    segmap = np.transpose(segmap, (1,2,0))
    pil_image=Image.fromarray(segmap)
    pil_image.show()
    