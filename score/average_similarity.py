class CumulativeScore:
    """
    实时累计平均相似度计算器。
    只保存累计和与帧数，内存占用极低。
    """
    def __init__(self):
        self.n = 0
        self.sum_score = 0.0
        self.scores = []  # 新增：保存所有分数

    def update(self, score):
        if score is not None:
            self.n += 1
            self.sum_score += score
            self.scores.append(score)  # 新增：追加分数

    @property
    def average(self):
        return self.sum_score / self.n if self.n > 0 else None
