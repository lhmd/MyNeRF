import os
import torch
import numpy as np
import json
from PIL import Image

# 上下平移矩阵
translate_positive_z = lambda z: torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, z],
    [0, 0, 0, 1]
], dtype=torch.float32)

# 绕x轴对世界坐标旋转矩阵,CCW意思是逆时针旋转
rotate_world_x_ccw = lambda phi: torch.tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

# 绕y轴对世界坐标逆时针旋转矩阵
rotate_world_y_ccw = lambda theta: torch.tensor([
    [np.cos(theta), 0, -np.sin(theta), 0],
    [0, 1, 0, 0],
    [np.sin(theta), 0, np.cos(theta), 0],
    [0, 0, 0, 1]
], dtype=torch.float32)

# 在相机坐标系中，x 轴是相机光轴的方向，y 轴是相机向右的方向，z 轴是相机向上的方向。
# 而在世界坐标系中，x 轴是世界坐标系的右方向，y 轴是世界坐标系的上方向，z 轴是世界坐标系的前方向。
# 因此，如果要将相机坐标系转换到世界坐标系中，需要将相机坐标系中的 y 轴和 z 轴进行交换，以实现坐标系的对齐。
# 在相机坐标系中，x 轴的方向指向相机的光轴，而且是从相机向外延伸的，
# 而在世界坐标系中，x 轴的方向是从世界坐标系的原点向右延伸的。
# 因此，如果只是交换相机坐标系中的 y 轴和 z 轴，坐标系的方向就会与世界坐标系的方向不同，
# 需要通过将相机坐标系中的 x 轴反向来调整坐标系的方向，使其与世界坐标系的方向一致。
change_worldCoordinate_yz_axis = torch.tensor([
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)
def pose_spherical(theta, phi, radius):
    """
    这个函数实现了一个从相机坐标系到世界坐标系的变换，它使用了一系列的矩阵变换来实现这个过程。
    首先，函数使用了一个平移变换 translate_positive_z(radius) 来将相机坐标系移动到世界坐标系的原点，其中 radius 是相机到原点的距离。
    然后，函数使用了一个绕 x 轴逆时针旋转的变换 rotate_worldCoordinate_x_axis_CCW(phi / 180. * np.pi) 来旋转相机坐标系，其中 phi 是相机绕 x 轴旋转的角度。
    接着，函数使用了一个绕 y 轴逆时针旋转的变换 rotate_worldCoordinate_y_axis_CCW(theta / 180. * np.pi) 来进一步旋转相机坐标系，其中 theta 是相机绕 y 轴旋转的角度。
    最后，函数使用了一个坐标轴变换 change_worldCoordinate_yz_axis 将新的坐标系与世界坐标系对齐，这样相机坐标系就变成了世界坐标系中的一个子坐标系。
    通过这些变换，函数可以将一个点从相机坐标系转换到世界坐标系中，从而实现了相机到世界的变换。
    :param theta: -180 -- +180，间隔为9
    :param phi: 固定值 -30
    :param radius: 固定值 4
    :return:
    """
    c2w = translate_positive_z(radius)
    # @是矩阵乘法
    c2w = rotate_world_x_ccw(phi / 180. * np.pi) @ c2w
    c2w = rotate_world_y_ccw(theta / 180. * np.pi) @ c2w
    c2w = change_worldCoordinate_yz_axis @ c2w
    return c2w

def load_blender_data(dirpath, half_res=False, testSkip=1,
                      renderSize=40, renderAngle=30.0):
    """
    输出：图像，每幅图像的相机世界转换矩阵，渲染时的pose，图片的宽高焦距，训练测试验证集的大小
    :param dirpath: 存放路径
    :param half_res: 是否降采样一半
    :param testSkip: test和val数据集，只会读取其中的一部分数据，跳着读取
    :param renderSize: 一圈多少帧，默认40
    :param renderAngle: 看的角度，固定30
    :return:
    Img=(400,H,W,4) where 4 is RGBA
    Pose=(400,4,4)
    RenderPose=(renderSize,4,4)
    index_split= 3 numpy array with 0~99,100~199,200~399
    each correspond to train,val,test idx
    """
    splits = ['train', 'val', 'test']
    jsons = {}
    for s in splits:
        with open(os.path.join(dirpath, 'transforms_{}.json'.format(s)), 'r') as f:
            jsons[s] = json.load(f)
    allImg = []
    allPose = []
    counts = [0]
    # 这里是把训练测试验证集分开
    for s in splits:
        if s == 'train' or testSkip == 0:
            skip = 1
        else:
            skip = testSkip

        jsonData = jsons[s]
        Imgs = []
        Poses = []
        # 这里是把每个集合中的各个图片分开
        for frame in jsonData['frames'][::skip]:
            # 将./去掉，方便后面合并路径
            file_path = frame['file_path'].replace('./', '')
            matrix = np.array(frame['transform_matrix'], dtype=np.float32)
            img = Image.open(os.path.join(dirpath, file_path + 'png'))
            if half_res:
                H, W = img.height, img.width
                H = H // 2
                W = W // 2
                # 这行代码使用Python的Pillow库中的resize()函数
                # 将一个名为img的图像对象重新调整为指定的高度H和宽度W，
                # 同时使用LANCZOS滤波器进行重采样（即重新采样并生成新的像素值）。
                # 具体地，resize()函数接受两个参数，即目标图像的新宽度和高度，
                # 以及一个可选的参数resample，用于指定重采样算法。在这里，
                # 我们使用Image.LANCZOS指定Lanczos滤波器，这是一种用于图像重采样的高质量滤波器。
                # 该滤波器使用像素周围的样本点进行插值，以生成高质量的缩放结果。
                img = img.resize((H, W), resample=Image.LANCZOS)
                Imgs.append(img)
                Poses.append(matrix)
            # counts[-1]代表的是倒数第一个counts的元素，比如counts[0]训练集个数为100，len为50，那么新加入的元素就是150
            counts.append(counts[-1] + len(Imgs))
            allImg.append(Imgs)
            allPose.append(Poses)
        # arange生成一组等间隔的数组,比如counts[1]=100, counts[2]=200,那么他会生成100-200步长为1的数组
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
        allImg = np.concatenate(allImg, axis=0)
        allPose = np.concatenate(allPose, axis=0)

        H, W = allImg[0].shape[:2]
        camera_angle_x = jsons['train']['camera_angle_x']
        focal = 0.5 *W / np.tan(0.5 * camera_angle_x)

        # 这行代码使用 PyTorch 框架实现了一个函数调用，
        # 其中包含一个 for 循环。该函数调用名为 pose_spherical，
        # 它的输入参数包括 theta、-renderAngle 和 4.0。
        # 在 for 循环中，theta 取值范围为从 -180 到 180，
        # 共 renderSize+1 个值，并取前 renderSize 个值，
        # 这些值会被用作 pose_spherical 函数的输入参数之一。
        # torch.stack 函数用于将 pose_spherical 函数的输出结果按照第一个维度拼接起来，
        # 形成一个张量。其中 dim=0 参数表示按照第一个维度进行拼接。
        # [:-1] 表示对一个列表或者数组进行切片操作，截取从第一个元素到倒数第二个元素（不包括最后一个元素）之间的所有元素，即去掉最后一个元素。
        render_poses = torch.stack([pose_spherical(theta, -renderAngle, 4.0)
                                    for theta in np.linspace(-180, 180, renderSize + 1)[:-1]], dim=0)
        return allImg, allPose, render_poses, [H, W, focal], i_split




