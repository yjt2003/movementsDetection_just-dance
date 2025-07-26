import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


#不同于欧氏距离是细化到每个点刻画对应点之间的差异，余弦向量法是在宏观上刻画两个动作的匹配程度

def compute_cosine_similarity(self, landmarks1, landmarks2):

    #注意注意，只有两个向量，也就是landmarks识别到的点数量相等的时候，才能进行匹配
    if len(landmarks1) != len(landmarks2):
        print("You must have the vectors of same length of vector")
        return None

    # 将 landmark 对象转换为扁平化的向量
    vec1 = np.array([[lm.x, lm.y] for lm in landmarks1]).flatten()
    vec2 = np.array([[lm.x, lm.y] for lm in landmarks2]).flatten()

    # 正规化（optional，但常见于 pose 比较任务中）
    vec1 = vec1 - np.mean(vec1)
    vec2 = vec2 - np.mean(vec2)

    # 使用 sklearn 的 cosine_similarity
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return similarity