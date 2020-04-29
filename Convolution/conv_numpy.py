'''
@Author: Fu guanyu
@time:   2020/04/29
@refe: https://github.com/yizt/numpy_neural_network/blob/master/2_1-numpy%E5%8D%B7%E7%A7%AF%E5%B1%82%E5%AE%9E%E7%8E%B0.ipynb
       https://blog.csdn.net/weixin_43217928/article/details/88597216
'''

import numpy as np

'''
标准的二维卷积操作：
inputs shape: [batch_size, H, W, C_in]
filters shape: [C_in, K, K, C_out]
outputs shape: [batch_size, H_n, W_n, C_out]
'''

def numpy_conv(input, filter, padding='VALID', stride=1):
    '''
    @ parameters
    :dtype input: numpy array, shape=[batch_size, H, W, C_in]
    :dtype filter: numpy array, shape=[C_in, K, K, C_out]
    :dtype b     : numpy array, shape=[C_out, ]
    :dtype padding: string, VALID or SAME
    :dtype stride:  constant

    :rtype: [batch_size, H_new, W_new, c_out]
    '''
    
    batch_size, H, W, C_in = input.shape
    C_in, k1, k2, C_out = filter.shape
    assert padding in ['SAME', 'VALID'], print("Choosing right padding model: SAME or VALID")
    assert (H - k1) % stride == 0, '步长不为1时，步长必须刚好能够整除'
    assert (W - k2) % stride == 0, '步长不为1时，步长必须刚好能够整除'

    if padding == 'SAME':
        p1 = ((H*stride - stride + k1)  - H) // 2
        p2 = ((W*stride - stride + k2)  - H) // 2
        input = np.lib.pad(
            array=input, 
            pad_width=[(0,0), (p1, p1), (p2, p2), (0,0)],
            mode='constant',
            constant_values=0)
    # 更新 H，W
    batch_size, H, W, C_in = input.shape
    C_in, k1, k2, C_out = filter.shape
    H_new, W_new = 1 + (H-k1) // stride, 1 + (W-k2) // stride
    result = np.zeros((batch_size, H_new, W_new, C_out))

    # 卷积核通过输入的每块区域，stride=1，注意输出坐标的起始位置
    for b in range(batch_size):
        for d in range(C_out):
            for h in range(0, stride, H - k1 + 1):
                for w in range(0, stride, W - k2 + 1):
                    # 池化大小的输入区域
                    cur_input = input[b, h:h+k1, w:w+k2, :]
                    # 与核进行乘法计算
                    cur_filter = filter[:, :, :, d].reshape([k1, k2, C_in])
                    cur_output = cur_input * cur_filter
                    # 在把所有值求和
                    conv_sum = np.sum(cur_output)
                    # 当前点的输出值
                    result[b, h, w, d] = conv_sum
    return result

input = np.zeros((32, 6, 6, 10))
filter = np.zeros((10, 2, 2, 66))
output = numpy_conv(input, filter, padding='VALID', stride=2)
print(output.shape)
