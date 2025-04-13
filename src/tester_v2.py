import torch
from model_operation import ModelOperator
from dataloader import GraphDataset, GCNRSTrainDataset
from torch.utils.data import DataLoader
from metrics_v2 import Metrics
import numpy as np
from tqdm import tqdm


class Tester(object):
    def __init__(self, flags_obj, model_operator: ModelOperator):
        super(Tester, self).__init__()
        self.flags_obj = flags_obj
        self.model_operator = model_operator
        self.Metrics = Metrics()
        self.results = {metric: 0.0 for metric in self.Metrics.metrics}

        self.init()

    def init(self):
        test_dataset = GCNRSTrainDataset(self.model_operator.dataset)
        self.dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.flags_obj.test_batch_size,
            shuffle=True,
            drop_last=False
        )

    def test(self, epoch):
        self.model_operator.load_model(epoch)
        with torch.no_grad():
            for batch_count, data in enumerate(tqdm(self.dataloader, desc='test epoch {}'.format(epoch))):
                batch_user, batch_test_positive_items, batch_train_positive_items = data

                rating = self.rating_after_filter(batch_user, batch_train_positive_items)

                _, batch_topk_items = torch.topk(rating, k=self.flags_obj.topk)
                batch_topk_items = batch_topk_items.cpu()
                # rating = rating.cpu().numpy()

                self.update_batch_results(batch_topk_items, batch_test_positive_items)
            self.average_results()

    def rating_after_filter(self, users, train_positive_items):
        """

        :param users:
        :param train_positive_items:
        :return: 过滤训练样本评分之后的预测评分
        """
        rating = self.model_operator.getUsersRating(users)
        exclude_index = []
        exclude_items = []
        for index, items in enumerate(train_positive_items):
            # train_positive_items是经过-1填充之后的张量，这里统计非负数据才是真正的训练集item
            real_items_num = np.count_nonzero(items >= 0)
            real_items = items[:real_items_num].numpy()

            exclude_index.extend([index] * real_items_num)
            exclude_items.extend(real_items)

        rating[exclude_index, exclude_items] = -(1 << 10)

        return rating

    def update_batch_results(self, batch_topk_items, batch_test_positive_items):
        """

        :param batch_topk_items:
        :param batch_test_positive_items:
        :return: 将每个batch的结果叠加到成员变量results
        """
        # for topk in self.flags_obj.topks:
        # real_topk_items = batch_topk_items[:, :self.flags_obj.topk]
        for topk_items, test_positive_items in zip(batch_topk_items, batch_test_positive_items):

            if np.count_nonzero(test_positive_items >= 0) > 0:  # 测试集中存在user的正样本情况
                self.Metrics.init_set(topk_items, test_positive_items)

                for metric in self.Metrics.metrics:
                    metric_func = self.Metrics.get_metrics(metric)
                    metric_result = metric_func()
                    self.results[metric] = self.results[metric] + metric_result

    def average_results(self):
        """

        :return: 将最后叠加的结果取平均
        """
        dataset = self.model_operator.dataset
        test_users = list(map(lambda x: x[0] if x[1] else None, enumerate(dataset.lil_test_record.rows)))

        for metric in self.Metrics.metrics:
            self.results[metric] = self.results[metric] / len(test_users)
