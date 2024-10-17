import numpy as np
from src.vis.anatomy_colors import AnatomyColors


def vis_teeth_cls(pcn, cls):
    colors = AnatomyColors()
    color = np.ones((pcn.shape[0], 4))
    color[:, 0:3] *= 0.5

    for tooth_id in np.unique(cls):
        tooth_id = int(tooth_id)
        color[cls == tooth_id, 0:3] = colors.get_tooth_color(tooth_id, True)

    return np.concatenate((pcn, color), axis=1)


def vis_teeth_seg(pcn, seg):#x,y,z, n_{x}, n_{y}, n_{z}, r,g,b,ref
    color = np.ones((pcn.shape[0], 4))
    color[:, 0:3] *= np.repeat(np.expand_dims(seg, 1), 3, 1)

    return np.concatenate((pcn, color), axis=1)


def vis_teeth_seg_cls(pcn, seg, cls, centroid=None):
    if type(cls) == np.ndarray:
        colors = AnatomyColors()
        color = np.ones((pcn.shape[0], 4))
        color[:, 0:3] *= 0.5
        for pred_cls in np.unique(cls):
            cls_color = colors.get_tooth_color(pred_cls, True)

            color[cls == pred_cls, 0:3] = cls_color

        if centroid is not None:
            color_centroid = np.array([[0, 0, 0, 1, 0, 0, 1]])
            centroid = np.concatenate((centroid, color_centroid), axis=1)
            return np.concatenate((np.concatenate((pcn, color), axis=1), centroid), axis=0)
        return np.concatenate((pcn, color), axis=1)
    else:
        colors = AnatomyColors()
        cls_color = colors.get_tooth_color(cls, True)
        color = np.ones((pcn.shape[0], 4))
        color[:, 0:3] *= 0.5

        for point_id in range(len(pcn)):
            conf = seg[point_id]
            color[point_id, 0:3] = cls_color * conf

        if centroid is not None:
            color_centroid = np.array([[0, 0, 0, 1, 0, 0, 1]])
            centroid = np.concatenate((centroid, color_centroid), axis=1)
            return np.concatenate((np.concatenate((pcn, color), axis=1), centroid), axis=0)
        return np.concatenate((pcn, color), axis=1)


def vis_teeth_heatmap(pcn, heatmap, centroid=None):
    low = np.array([0, 0, 1, 1])
    high = np.array([1, 0, 0, 1])
    color = np.ones((pcn.shape[0], 4))

    for point_id in range(len(pcn)):
        conf = heatmap[point_id]
        color[point_id] = high * conf + low * (1 - conf)

    if centroid is not None:
        color_centroid = np.array([[0, 0, 0, 1, 0, 0, 1]])
        centroid = np.concatenate((centroid, color_centroid), axis=1)
        return np.concatenate((np.concatenate((pcn, color), axis=1), centroid), axis=0)
    return np.concatenate((pcn, color), axis=1)


def vis_teeth_centroids(pcn, centroids):
    color = np.ones((pcn.shape[0], 4))
    color[:, 0:3] = color[:, 0:3] * 0.5
    centroids_color = np.ones((centroids.shape[0], 7))
    centroids_color[:, 4:6] = 0
    return np.concatenate((
        np.concatenate((pcn, color), axis=1),
        np.concatenate((centroids, centroids_color), axis=1)
    ), axis=0)


if __name__ == '__main__':
    from src.utils.input import read_obj
    from src.utils.output import write_obj
    import json
    verts, faces = read_obj('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/half-broken/ZKJEPFDD_upper.obj', False)
    with open('/home/zsj/Downloads/dental-labels.json', 'r') as fp:
        labels = json.load(fp)
        labels = labels['labels']
    colors = np.ones(verts.shape) * 200
    ac = AnatomyColors()
    for i in range(len(labels)):
        colors[i] = ac.get_tooth_district_color(labels[i])
    verts = np.concatenate((verts, colors), axis=-1)
    write_obj('/home/zsj/vis.obj', verts, faces, None)
