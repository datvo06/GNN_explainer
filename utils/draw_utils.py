from __future__ import print_function, unicode_literals
from PIL import ImageFont, ImageDraw, Image
import cv2

font = ImageFont.truetype(fontpath)

def get_colors_list_boxs():
    """
    B, G, R for openCV
    """
    return [(255, 0, 0), (0, 255, 0), (127, 127, 0)]


def get_colors_list_edges():
    return [(127, 0, 127), (0, 0, 0), (0, 127, 127)]


def draw_bbox(img, x, y, w, h, color, thickness=1):
    """
    This return the drawn image
    """
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    return cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)


def draw_text(img, x, y, w, h, text, default_font_scale=10):
    """
    This will never be called here, it will be call in the outer folder instead
    """
    fontpath = "./simsun.ttc"



def get_pseudo_text_bow_repr(bows, word_list):
    return str([word_list[i] for i in bows if i != 0])


def draw_arrow(img, x1, y1, x2, y2):
    pass


def visualize_graph(list_bows, list_positions,
                    adj_mats, word_list=None, is_text=False):
    """
    Args:
        list_bows: the bag of words sets
        list_positions: list of (x, y, w, h)
        adj_mats: matrix of NxNxE (E is number of edge types)
        word_list: the list of word (if we were to visualize bows)
        is_text: if the bows is texts
    """
