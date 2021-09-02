import cv2
import click
import numpy as np 
import matplotlib.pyplot as plt

@click.command()
@click.argument('input_image', type=click.Path(exists=True))
@click.option('--threshold', default=1.12, show_default=True, type=float, metavar='<float>',
    help = 'specify the threshold for green index')
@click.option('--border', nargs=4, show_default=True, default=(180,1750,850,1770), type=(int, int, int, int), 
    help = 'specify the border of the frame following upper, bottom, left, right')
def SegPlant(input_image, threshold, border):
    """whole plant segmentation for RGB image"""
    out_thresh = input_image.replace('.png', '.thresh.png')
    out_contour = input_image.replace('.png', '.contour.png')
    out_hull = input_image.replace('.png', '.hull.png')

    img = cv2.imread(input_image)
    img = np.where(img==0, 1, img)
    img = img.astype(np.float)
    img_r, img_g, img_b = img[:,:,0],img[:,:,1],img[:,:,2]
    g_idx = 2 * img_g/(img_r + img_b)
    thresh1 = np.where(g_idx > threshold, g_idx, 0)
    upper, bottom, left, right = border
    thresh1[0:upper, :] = 0
    thresh1[bottom:, :] = 0
    thresh1[:, 0:left] = 0
    thresh1[:, right:] = 0

    blur = cv2.GaussianBlur(thresh1, (5,5), 0)
    __, thresh2 = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
    cv2.imwrite(out_thresh, thresh2)
    thresh2 = thresh2.astype(np.uint8)
    __,contours,__ = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imwrite(out_contour, img)

    hull = [cv2.convexHull(contours[i]) for i in range(len(contours))]
    cv2.drawContours(img, hull, -1, (0,0,255), 3)
    cv2.imwrite(out_hull, img)

if __name__ == '__main__':
    SegPlant()
