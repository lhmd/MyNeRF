import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, depth=8, hidden_unit=256, position_ch=3,
                 direction_ch=3, output_ch=4, skip_connection=[4], use_viewdirs=True):
        """
        输入：编码过的位置和视角坐标
        输出：output[...,0:3] 是颜色RGB值， output[...,3]是 sigma
        :param depth: 神经网络输出前的深度，有八层，然后输出sigma
        :param hidden_unit: 中间的连接层宽度
        :param position_ch: gamma(x)的大小，就是xyz的大小
        :param direction_ch: gamma(d)的大小，就是视角二维向量转换过来的的三维向量
        :param output_ch: 视角不存在的时候这个值才有意义，一般用不到
        :param skip_connection: 跳过的那一层
        :param use_viewdirs: 视角不存在的时候是FALSE
        """

        super(NeRF, self).__init__()
        self.position_ch = position_ch
        self.direction_ch = direction_ch
        self.skip_connection = skip_connection
        self.output_ch = output_ch
        self.use_viewdirs = use_viewdirs
        self.pts_linears = nn.ModuleList([nn.Linear(position_ch, hidden_unit)] +
                                         [
                                             nn.Linear(hidden_unit, hidden_unit)
                                             if i not in skip_connection
                                             else nn.Linear(hidden_unit + position_ch, hidden_unit)
                                             for i in range(depth - 1)
                                         ])
        if self.use_viewdirs:
            self.sigma_layer = nn.Linear(hidden_unit, 1)
            self.feature_linear = nn.Linear(hidden_unit, hidden_unit)
            # 地板除，//代表除完之后只取整数部分
            self.view_linear = nn.Linear(hidden_unit + direction_ch, hidden_unit // 2)
            self.color_layer = nn.Linear(hidden_unit // 2, 3)
        else:
            self.output_linear = nn.Linear(hidden_unit, output_ch)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        # dim=-1代表按照最深的维度进行拆分
        input_pos, input_dir = torch.split(inputs, [self.position_ch, self.direction_ch], dim=-1)
        x = input_pos
        # enumerate的返回值是序号和数据(在这里是神经网络的各个层)
        for idx, layer in enumerate(self.pts_linears):
            # 先经过神经网络的连接层，然后使用relu进行非线性处理
            x = self.relu(layer(x))
            if idx in self.skip_connection:
                x = torch.cat([input_pos, x], dim=-1)

        if self.use_viewdirs:
            sigma = self.sigma_layer(x)
            # 原始论文中橙色的箭头有一个隐藏层
            x = self.feature_linear(x)
            x = self.view_linear(torch.cat([x, input_dir], dim=-1))
            x = self.relu()
            color = self.color_layer(x)
            outputs = torch.cat([color, sigma], dim=-1)
        else:
            outputs = self.output_linear(x)
        return outputs

