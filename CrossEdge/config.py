node_dict_path = "../data/la/node_dict111.txt"
edge_file_path = "../data/la/la-edges.txt"
embed_init = "../data/la/crossmap.emb"
model_log = "./log/"
write_file_path = "../data/la/embedding/crossedge-epoch"

# Model parameters
embed_dim = 300
neg_num = 100

# training parameters
load_model = False
lr_gen = 1e-3
max_epochs = 100
train_type = ['tw','tl', 'wl', 'wt', 'lt', 'lw']
batch_size = 64

# testing parameters
epoch_test = 10
result_pre = "../data/la/result/epoch-"
test_data = "../data/la/la-test.data"
crossmap = "../data/la/la-pickled.model"
predict_type = ['w', 'l', 't']
node_dict = "../data/la/node_dict111.txt"