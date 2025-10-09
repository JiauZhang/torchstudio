import numpy as np

def softmax(x, axis):
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def tiled_softmax(x, y, axis):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = np.exp(x_shifted)
    x_sum = np.sum(exp_x, axis=axis, keepdims=True)

    y_max = np.max(y, axis=axis, keepdims=True)
    y_shifted = y - y_max
    exp_y = np.exp(y_shifted)
    y_sum = np.sum(exp_y, axis=axis, keepdims=True)

    xy_max = np.maximum(x_max, y_max)
    exp_x_xy_max = np.exp(x_max - xy_max)
    exp_y_xy_max = np.exp(y_max - xy_max)
    xy_sum = exp_x_xy_max * x_sum + exp_y_xy_max * y_sum
    o = np.concat((exp_x_xy_max * exp_x, exp_y_xy_max * exp_y), axis=axis) / xy_sum
    return o

def tiled_attention(q, k, v, output, m_base, norm_base):
    # q shape: [B_r, d], k shape: [B_c, d]
    s = q @ k.T
    m = np.max(s, axis=1, keepdims=True)
    m_new = np.maximum(m, m_base)
    p = np.exp(s - m)
    norm = np.sum(p, axis=1, keepdims=True)
    exp_m_m_new = np.exp(m - m_new)
    exp_m_base_m_new = np.exp(m_base - m_new)
    norm_new = exp_m_base_m_new * norm_base + exp_m_m_new * norm
    output =  (norm_base * exp_m_base_m_new * output + exp_m_m_new * p @ v) / norm_new
    return output, m_new, norm_new

def full_attention(q, k, v):
    output = np.zeros_like(q)
    norm_base = np.zeros((q.shape[0], 1))
    m_base = norm_base - np.inf
    output = tiled_attention(q, k, v, output, m_base, norm_base)[0]
    return output

def __fetch_block(data, top, left, height, width):
    bottom = min(top + height, data.shape[0])
    right = min(left + width, data.shape[1])
    return data[top:bottom, left:right]

def flash_attention(q, k, v, M):
    N, d = q.shape
    B_c = int(np.ceil(M / 4 / d))
    B_r = int(min(np.ceil(M / 4 / d).item(), d))
    T_r = int(np.ceil(N / B_r))
    T_c = int(np.ceil(N / B_c))

    output = np.zeros_like(q)
    norm_base = np.zeros((q.shape[0], 1))
    m_base = norm_base - np.inf

    for j in range(T_c):
        # [B_c, d]
        K_j = __fetch_block(k, j * B_c, 0, B_c, d)
        V_j = __fetch_block(v, j * B_c, 0, B_c, d)
        for i in range(T_r):
            # [B_r, d]
            Q_i = __fetch_block(q, i * B_r, 0, B_r, d)
            O_i = __fetch_block(output, i * B_r, 0, B_r, d)
            # [B_r, 1]
            norm_i = __fetch_block(norm_base, i * B_r, 0, B_r, 1)
            m_i = __fetch_block(m_base, i * B_r, 0, B_r, 1)
            O_i[...], m_i[...], norm_i[...] = tiled_attention(Q_i, K_j, V_j, O_i, m_i, norm_i)

    return output
