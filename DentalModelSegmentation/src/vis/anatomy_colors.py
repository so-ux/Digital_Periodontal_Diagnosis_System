import os
import numpy as np


class AnatomyColors:
    def __init__(self):
        self.tooth_colors = np.loadtxt(os.path.join(os.path.dirname(__file__), 'tooth_rgb.txt'), dtype=int)
        self.anatomy_colors = np.loadtxt(os.path.join(os.path.dirname(__file__), 'anatomy_rgb.txt'), dtype=int)

    def get_color(self, i, norm=False):
        """
        Get generic anatomy color by index in RGB format

        See https://www.slicer.org/wiki/Slicer3:2010_GenericAnatomyColors for a full table

        :param i: Index, 0 is background (#000000)
        :type i: int
        :param norm: The color form. If true, the range is 0 ~ 255, else 0.0 ~ 1.0. <br> Default is <code>False</code>.
        :type norm: bool
        :return: A 3-dimensional NumPy array in order of [R,G,B].
        :rtype: ndarray
        """
        if norm:
            return self.anatomy_colors[i] / 255.0
        else:
            return self.anatomy_colors[i]

    def get_tooth_color(self, i, norm=False):
        if i == 0:
            return np.array([234, 234, 234]) / 255 if norm else np.array([234, 234, 234])
        if norm:
            return self.tooth_colors[(i - 1) % 8 + 1] / 255.0
        else:
            return self.tooth_colors[(i - 1) % 8 + 1]

    def get_tooth_district_color(self, i, norm=False):
        """
        获取分区牙齿颜色

        1X: 蓝；2X：红；3X：绿；4X：粉

        :param i: FDI标准的牙齿编号
        :type i: int
        :param norm: 是否归一化[0, 1]
        :type norm: bool
        :return:
        :rtype:
        """
        if i == 0:
            return np.array([234, 234, 234]) / 255 if norm else np.array([234, 234, 234])
        base_colors = np.array([
            [57, 74, 244],
            [255, 75, 75],
            [55, 158, 50],
            [223, 91, 236]
        ])
        base_colors = base_colors / 255 if norm else base_colors
        district = i // 10 - 1
        return np.asarray(base_colors[district] * 0.2 + 0.8 * self.get_tooth_color(i % 10, norm), dtype=np.int32)


if __name__ == '__main__':
    colors = AnatomyColors()
    for i in range(1, 5):
        print('<script>function rgbToHex(r, g, b) { return "#" + ( ( 1 << 24) + (r << 16) + (g << 8) + b).toString ( 16 ).slice ( 1 ); }</script>')
        print('<div style="display: flex">')
        for j in range(1, 9):
            c = colors.get_tooth_district_color(i * 10 + j)
            print(f'<div style="width: 160px; padding: 16px; background: rgb({c[0]}, {c[1]}, {c[2]})"><div>{i}{j}</div>'
                  f'<div>RGB {c[0]} {c[1]} {c[2]}</div>'
                  f'<div id="c{i}{j}"></div>'
                  f'<script>document.getElementById("c{i}{j}").innerHTML = rgbToHex({c[0]}, {c[1]}, {c[2]})</script>'
                  f'</div>')
        print('</div>')

    # print('<br>')
    # for i in range(1, 5):
    #     print('<script>function rgbToHex(r, g, b) { return "#" + ( ( 1 << 24) + (r << 16) + (g << 8) + b).toString ( 16 ).slice ( 1 ); }</script>')
    #     print('<div style="display: flex">')
    #     for j in range(1, 9):
    #         c = colors.get_color((i - 1) * 8 + j)
    #         print(f'<div style="width: 160px; padding: 16px; background: rgb({c[0]}, {c[1]}, {c[2]})"><div>{i}{j}</div>'
    #               f'<div>RGB {c[0]} {c[1]} {c[2]}</div>'
    #               f'<div id="c{i}{j}"></div>'
    #               f'<script>document.getElementById("c{i}{j}").innerHTML = rgbToHex({c[0]}, {c[1]}, {c[2]})</script>'
    #               f'</div>')
    #     print('</div>')
