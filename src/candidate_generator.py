import faiss
import numpy as np


class TopkSearchGenerator(object):
    def __init__(self, items_embedding, flags_obj):
        self.items_embedding = items_embedding
        self.embedding_size = items_embedding.shape[1]
        self.flags_obj = flags_obj
        self.index = None
        self.init_index()

    def init_index(self):
        # 设置index索引库
        self.index = faiss.IndexFlatIP(self.embedding_size)  # 直接用IndexFlatIP索引精度不高，所以后续增加归一和正则
        items_embedding_ = self.items_embedding.copy() / np.linalg.norm(self.items_embedding)  # 归一化
        faiss.normalize_L2(items_embedding_)  # 正则化
        self.index.add(items_embedding_)  # 添加items索引库
        # self.index.add(self.items_embedding)  # 添加items索引库

        if self.flags_obj.faiss_use_gpu:
            provider = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(provider, self.flags_obj.faiss_gpu_id, self.index)

    # 返回推荐topk的item索引
    def generate(self, users_embedding, topk):
        if len(users_embedding.shape) == 2:
            D, I = self.index.search(users_embedding, topk)
        elif len(users_embedding.shape) == 1:
            users_embedding = np.expand_dims(users_embedding, axis=0)
            D, I = self.index.search(users_embedding, topk)
        else:
            raise Exception
        return I

    def generate_with_distance(self, users_embedding, topk):
        D, I = self.index.search(users_embedding, topk)

        return D, I
