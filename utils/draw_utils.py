"""
IMPORTANCE: bbox format is (x, y, w, h)
"""
from __future__ import print_function, unicode_literals
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os
import cv2


currdir = os.path.dirname(os.path.abspath(__file__))

for i in range(1, 12, 2):
    fonts = ImageFont.truetype(
        os.path.join(currdir, "arial-unicode-ms.ttf"), i)


def get_used_font(top_height):
    i = 0
    while(top_height < fonts[i].font.height):
        i += 1
    return fonts[i]


def get_colors_list_boxs():
    """
    B, G, R for openCV
    """
    return [(255, 0, 0), (0, 255, 0), (127, 127, 0)]


def get_colors_list_edges():
    return [(127, 0, 127), (0, 0, 0), (0, 127, 127)]


def draw_bbox(img, x, y, w, h, color, thickness=5):
    """
    This return the drawn image
    """
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    return cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)


def draw_node_bbox(img, bbox, label, importance=1.0):
    """
    For each node, denote significant by bbox
    """
    colors = get_colors_list_boxs()
    return draw_bbox(img, bbox[0], bbox[1], bbox[2], bbox[3],
                     colors[label % len(colors)],
                     min(max(int(5*importance), 1.0), 5))


def draw_nodes_bboxs(img, list_bboxs, labels, imporantces):
    for i, bbox in enumerate(list_bboxs):
        img = draw_node_bbox(img, bbox, labels[i], imporantces[i])
    return img


def draw_text(img, x, y, w, h, text, color=(0, 0, 0),
              importances=None):
    """
    This will never be called here, it will be call in the outer folder instead
    TODO: Testing
    """
    font = get_used_font(h)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    charwidth = font.font.width
    if importances is None:
        draw.text(
            (x, y+h), text, font=get_used_font(),
            fill=(color))
    else:
        for i, character in enumerate(text):
            draw.text(
                (x+i*charwidth+1, y+h),
                character, font=get_used_font(),
                fill=(
                    tuple(
                        [int(color_elem*0.5*(1-importances[i]))
                         for color_elem in color])
                ),
                )
    return np.array(pil_img)


def draw_node_text_importances(img, list_bboxs, list_texts,
                               bow_dict,
                               bow_importances):

    get_mapping = (lambda c: c)\
        if bow_dict is None else (lambda c: bow_dict[c])
    for i, bbox in enumerate(list_bboxs):
        img = draw_text(
            img, bbox[0], bbox[1], bbox[2], bbox[3],
            list_texts[i],
            importances=[bow_importances[i][get_mapping(c)]
                         for c in list_texts])
    return img


def get_pseudo_text_bow_repr(bows, word_list):
    return str([word_list[i] for i in bows if i != 0])


def draw_arrow(img, x1, y1, x2, y2, color, thickness=1):
    return cv2.arrowedLine(img, (int(x1), int(y1)),
                           (int(x2), int(y2)), color, thickness)


def draw_node_position_feature_importances(img, bbox, node_pos_importances):
    # feature order: topleft, top right, lower right, lowerleft
    x, y, w, h = bbox
    pts = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    for i, pt in enumerate(pts):
        cv2.circle(img, pt, 5*node_pos_importances[i], (0, 255, 0), -1)


def draw_position_feature_importances(img, list_bboxs, position_importances):
    """
    Denote significant of features by radius of filled circles
    """
    img = img.copy()
    for i, bbox in enumerate(list_bboxs):
        draw_node_position_feature_importances(img,
                                               bbox,
                                               position_importances[i])
    return img


def draw_nodes(img, list_texts, list_bboxs, node_labels,
               node_importances,
               position_importances,
               bow_importances,
               bow_dict):
    """
    Args:
        Node importances: N
        position_importances: Nx4
        bow importances: NxBoW
        bow_dict: dict mapping from char to feat index
    """
    # First, draw the nodes based on the importances
    img = draw_nodes_bboxs(img, list_bboxs, node_labels, node_importances)

    # Then, draw each nodes's positional importances
    img = draw_node_position_feature_importances(
        img, list_bboxs, position_importances)
    # Finally, draw each node's text with importances
    draw_node_text_importances(img, list_bboxs, list_texts,
                               bow_dict, bow_importances)


def visualize_graph(list_bows, list_positions,
                    adj_mats, node_labels,
                    node_importances,
                    position_importances,
                    bow_importances,
                    adj_importances,
                    orig_img=None,
                    word_list=None, is_text=False):
    """
    Args:
        list_bows: the bag of words sets
        list_positions: list of (x, y, w, h)
        adj_mats: matrix of NxNxE (E is number of edge types)
        word_list: the list of word (if we were to visualize bows)
        is_text: if the bows is texts
    """
    # First, get the texts for all bows
    list_texts = list_bows[:]
    bow_dict = dict([(word, i) for i, word in enumerate(word_list)]
                    ) if word_list is not None else None
    if not is_text and word_list is not None:
        list_texts = [get_pseudo_text_bow_repr(bow, word_list)
                      for bow in list_bows]
    max_x = np.max(list_positions[:, 0] + list_positions[:, 2])
    max_y = np.max(list_positions[:, 1] + list_positions[:, 3])
    if orig_img is None:
        img = np.zeros((max_y, max_x), dtype=np.uint8)
    else:
        img = orig_img.copy()
    # first, draw the nodes
    draw_nodes(img, list_texts, list_positions,
               node_labels, node_importances, position_importances,
               bow_importances, bow_dict)
    return img
