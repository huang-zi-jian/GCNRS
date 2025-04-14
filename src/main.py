from absl import app, flags, logging
from trainer import TrainManager
from config import logging_config
from visualization import t_SNE, t_SNE_v2
import scipy.sparse as sp
from dataloader import GraphDataset
from MIA_SP_v8 import MIA
import torch.nn.functional as func
import numpy as np
# from AED_SP_wo_str import AED
import torch

# set_seed(2020)

flags_obj = flags.FLAGS

flags.DEFINE_integer("batch_size", 2048, "Train batch size.")
flags.DEFINE_integer("test_batch_size", 256, "Test batch size.")
flags.DEFINE_integer("epochs", 1000, "The number of epoch for training.")
flags.DEFINE_integer("warmup_steps", 150, "The warmup steps.")
flags.DEFINE_integer("embedding_dim", 64, "Embedding dimension for embedding based models.")
flags.DEFINE_integer("faiss_gpu_id", 0, "GPU ID for faiss search.")
flags.DEFINE_integer("n_layers", 3, "The layer number of lightGCN.")
flags.DEFINE_integer("q", 5, "SVD q.")
flags.DEFINE_integer("pool_num", 2, "Negative samples number.")
# flags.DEFINE_integer("topk", 20, "Top k for testing recommendation performance.")
flags.DEFINE_multi_integer("topks", [20], "Top k for testing recommendation performance.")
flags.DEFINE_float("lr", 0.001, "Learning rate.")
flags.DEFINE_float("temp", 0.5, "Weight to balance prior.")
flags.DEFINE_float("prior_weight", 0.01, "Weight to balance prior.")
flags.DEFINE_float("static_prob", 0.98, "Rate for dropout.")
flags.DEFINE_float("cl_weight", 0.001, "Weight of cl loss.")
flags.DEFINE_float("weight_decay", 1e-6, "Weight decay of optimizer.")
flags.DEFINE_float("str_weight", 0.01, "Weight of structure loss.")
flags.DEFINE_float("margin", 0.1, "Margin of structure loss.")
flags.DEFINE_float("alpha", 0.5, "alpha parameter.")
flags.DEFINE_string("output", '/Users/Master/GCNRS/output-1', "Folder for experiment result.")
# flags.DEFINE_string("exp_name", "experiment", "Experiment name.")
flags.DEFINE_string("dataset", "/Users/Master/GCNRS/datasets/", "Folder for dataset.")
flags.DEFINE_enum("device", "cpu", ['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], 'Device setting for training.')
flags.DEFINE_enum("dataset_name", "coat",
                  ["Ciao", "coat", "book-crossing", "lastfm", "ml1m", "yahoo", "Ciao_bias", "lastfm_bias",
                   "ml1m_bias", "book_bias", "coat_bias", "yahoo_bias"],
                  "Name of dataset.")
flags.DEFINE_enum("model_name", "GSCL", ['lightGCL', 'lightGCN', 'EXMF', 'FAWMF', 'MIA', 'GSCL', 'WMF'],
                  'Model for training.')
flags.DEFINE_enum("discrepancy_loss", "dCor", ['L1', 'L2', 'dCor'], 'Discrepancy loss function.')
flags.DEFINE_enum("watch_metric", "recall", ['precision', 'recall', 'hit_ratio', 'ndcg'],
                  "Metric for scheduler's step.")
flags.DEFINE_enum("data_source", "valid", ['test', 'valid'], 'Which dataset to test.')
flags.DEFINE_multi_string('metrics', ['precision', 'recall', 'hit_ratio', 'ndcg'], 'Metrics list.')
flags.DEFINE_bool("str_intervention", True, "Whether intervention for structure or not.")
flags.DEFINE_bool("pre_intervention", False, "Whether intervention for preference or not.")
flags.DEFINE_bool("adj_split", False, "Whether split matrix or not.")
flags.DEFINE_bool("dropout", False, "Whether drop graph or not.")
flags.DEFINE_bool("faiss_use_gpu", False, "Use GPU or not for faiss search.")


# lightGCN遵循NGCF的惯例，将数据集8:2划分为训练集和测试集，通过随机选择每个用户80%的历史交互
# logging.set_verbosity(logging.DEBUG)  # 将log打印的最低级别设置为debug
# logging.use_absl_handler()
# logging.get_absl_handler().setFormatter(None)


def main(argv):
    # workspace在调试的时候可以自己手动设置
    # workspace = 'D:/project/GCNRS/output/VAEDICE_2023-04-20-16-51'

    config = logging_config(flags_obj)
    config.set_train_logging()
    trainer = TrainManager.get_trainer(flags_obj, config.workspace)
    trainer.train()

    # model_path = 'D:\project\AED\output\Ciao_AED_2024-04-21-17-39\ckpt\epoch_999.pth'

    # model_path = r'D:\project\AED\output\Ciao_AED_2024-04-21-09-23\ckpt\epoch_699.pth'  # w/o str
    # model_path = r'D:\project\AED\output\Ciao_AED_2024-04-21-17-36\ckpt\epoch_699.pth'  # DSE
    # model_path = r'D:\project\AED\output\Ciao_AED_2024-04-24-08-35\ckpt\epoch_0.pth'  # w/o svd

    # model = AED(GraphDataset(flags_obj))
    # model.load_state_dict(torch.load(model_path, map_location=model.flags_obj.device))
    # users_preference, items_preference = model.pGcn()
    # users_structure, items_structure = model.structure()
    #
    # users_exposure_embed = users_structure
    # items_exposure_embed = items_structure
    #
    # users_embed = torch.cat((users_exposure_embed, users_preference), dim=-1)
    # items_embed = torch.cat((items_exposure_embed, items_preference), dim=-1)
    # click_rating = func.sigmoid(torch.matmul(users_embed[18], items_embed.t()))
    #
    # train_csr_record = sp.load_npz(r'D:\project\AED\dataset\Ciao\train_csr_record.npz')
    # train_csr_record = train_csr_record.astype(np.bool).astype(np.int)  # 有数据集存在重复项
    #
    # mask = train_csr_record[18].toarray()
    # mask = torch.Tensor(mask).to(flags_obj.device)
    # predictions = click_rating * (1 - mask) - 1e3 * mask
    # predictions = predictions.argsort(descending=True)
    #
    # predictions = np.array(predictions.cpu())[:, :20]
    #
    # print()
    # users_embed = model.user_embedding[18]
    # items_embed = model.item_embedding
    # rating = func.sigmoid(torch.matmul(users_embed, items_embed.t()))
    # t_SNE_v2(model)
    # t_SNE(model)


if __name__ == '__main__':
    app.run(main)
