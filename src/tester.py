import torch
from model_operation import ModelOperator
from dataloader import GraphDataset, minibatch
from metrics import RecallPrecision_ATk, NDCGatK_r, getLabel
import numpy as np


class Tester(object):
    def __init__(self, flags_obj, model_operator: ModelOperator):
        super(Tester, self).__init__()
        self.flags_obj = flags_obj
        self.model_operator = model_operator

    def test_one_batch(self, x):
        topk_items = x[0].numpy()
        groundTrue = x[1]
        hit = getLabel(groundTrue, topk_items)
        pre, recall, ndcg = [], [], []
        for k in self.flags_obj.topks:
            ret = RecallPrecision_ATk(groundTrue, hit, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(NDCGatK_r(groundTrue, hit, k))
        return {'recall': np.array(recall),
                'precision': np.array(pre),
                'ndcg': np.array(ndcg)}

    def test(self):
        dataset = self.model_operator.dataset
        results = {'precision': np.zeros(len(self.flags_obj.topks)),
                   'recall': np.zeros(len(self.flags_obj.topks)),
                   'ndcg': np.zeros(len(self.flags_obj.topks))}

        with torch.no_grad():
            # users = list(range(dataset.num_users))
            # 删除测试集中不存在交互的user
            test_users = list(map(lambda x: x[0] if x[1] else None, enumerate(dataset.lil_test_record.rows)))

            rating_list = []
            groundTrue_list = []

            for batch_users in minibatch(test_users, batch_size=self.flags_obj.test_batch_size):
                train_positives_list = []  # 训练集交互项目
                test_positives_list = []  # 测试集交互项目
                for user in batch_users:
                    user_test_positives = dataset.lil_test_record.rows[user]
                    user_train_positives = dataset.lil_train_record.rows[user]

                    train_positives_list.append(user_train_positives)
                    test_positives_list.append(user_test_positives)

                batch_users = torch.Tensor(batch_users).long()
                rating = self.model_operator.getUsersRating(batch_users)
                exclude_index = []
                exclude_items = []
                for user, items in enumerate(train_positives_list):
                    exclude_index.extend([user] * len(items))
                    exclude_items.extend(items)

                rating[exclude_index, exclude_items] = -(1 << 10)
                _, rating_k = torch.topk(rating, k=self.flags_obj.topks)
                # rating = rating.cpu().numpy()

                del rating
                # users_list.append(batch_users)
                rating_list.append(rating_k.cpu())
                groundTrue_list.append(test_positives_list)

            X = zip(rating_list, groundTrue_list)
            pre_results = []
            for x in X:
                pre_results.append(self.test_one_batch(x))

            # scale = float(batch_size / len(users))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(test_users))  # todo: 这里users应该改成测试集user数量
            results['precision'] /= float(len(test_users))
            results['ndcg'] /= float(len(test_users))

            return results
