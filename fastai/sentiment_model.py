from fastai.torch_core import *
from fastai.basic_train import *
from fastai.callbacks import *
from fastai.data_block import CategoryList
from fastai.basic_data import *
from fastai.datasets import *
from fastai.metrics import accuracy
from fastai.train import GradientClipping
from fastai.layers import *
from fastai.text.models import *
from fastai.text.transform import *
from fastai.text.data import *
from fastai.text import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score

class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).
        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort
        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)

class DynamicLSTM(nn.Module):
    '''
    LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, lenght...).
    '''

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        '''
        sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort
        '''
        '''sort'''
        x_sort_idx = torch.sort(x_len, descending=True)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        '''pack'''
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        ''' process '''
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        '''unsort'''
        ht = ht[:, x_unsort_idx]
        if self.only_use_last_hidden_state:
            return ht
        else:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
            if self.batch_first:
                out = out[x_unsort_idx]
            else:
                out = out[:, x_unsort_idx]
            if self.rnn_type == 'LSTM':
                ct = ct[:, x_unsort_idx]
            return out, (ht, ct)

class SqueezeEmbedding(nn.Module):
    '''
    Squeeze sequence embedding length to the longest one in the batch
    '''
    def __init__(self, batch_first=True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def forward(self, x, x_len):
        '''
        sequence -> sort -> pad and pack -> unpack -> unsort
        '''
        '''sort'''
        x_sort_idx = torch.sort(x_len, descending=True)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        '''pack'''
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        '''unpack'''
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(x_emb_p, batch_first=self.batch_first)
        if self.batch_first:
            out = out[x_unsort_idx]
        else:
            out = out[:, x_unsort_idx]
        return out

class SoftAttention(nn.Module):
    '''
    Attention Mechanism for ATAE-LSTM
    '''
    def __init__(self, hidden_dim, embed_dim):
        super(SoftAttention, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.w_h = nn.Linear(hidden_dim, hidden_dim, bias=False).cuda()
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False).cuda()
        self.w_p = nn.Linear(hidden_dim, hidden_dim, bias=False).cuda()
        self.w_x = nn.Linear(hidden_dim, hidden_dim, bias=False).cuda()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim + embed_dim)).cuda()
        self.fc_dp = nn.Dropout(0.2)  # fc dropout
    def forward(self, h, aspect):
        hx =  self.w_h(h)
        vx = self.w_v(aspect)
        hv = F.tanh(torch.cat((hx, vx), dim=-1))
        ax = torch.unsqueeze(F.softmax(torch.matmul(hv, self.weight), dim=-1), dim=1)
        rx = torch.squeeze(torch.bmm(ax, h), dim=1)
        hn = h[:, -1, :]
        hs = F.tanh(self.w_p(rx) + self.w_x(hn))
        # hs = self.fc_dp(hs)
        return rx, hs

class ATAE_LSTM(nn.Module):
    ''' Attention-based LSTM with Aspect Embedding '''
    def __init__(self, embedding_weights):
        super(ATAE_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_weights, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(400+400, 400, num_layers=1, batch_first=True)
        self.attention = SoftAttention(400, 400)
        self.dense = nn.Linear(400, 3)

    def forward(self, inputs):
        text, aspect_text = inputs[:,0,:], inputs[:,1,:]
        x_len = torch.sum(text != 1, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_text != 1, dim=-1).float()

        x = self.embed(text)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_text)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, _ = self.lstm(x, x_len)
        hs = self.attention(h, aspect)
        out = self.dense(hs)
        return out

class AlignmentMatrix(nn.Module):
    def __init__(self):
        super(AlignmentMatrix, self).__init__()
        self.w_u = nn.Parameter(torch.Tensor(3*400, 1))

    def forward(self, batch_size, ctx, asp):
        ctx_len = ctx.size(1)
        asp_len = asp.size(1)
        alignment_mat = torch.zeros(batch_size , ctx_len, asp_len).cuda()
        ctx_chunks = ctx.chunk(ctx_len, dim=1)
        asp_chunks = asp.chunk(asp_len, dim=1)
        for i, ctx_chunk in enumerate(ctx_chunks):
            for j, asp_chunk in enumerate(asp_chunks):
                feat = torch.cat([ctx_chunk, asp_chunk, ctx_chunk*asp_chunk], dim=2) # batch_size x 1 x 6*hidden_dim
                alignment_mat[:, i, j] = feat.matmul(self.w_u.expand(batch_size, -1, -1)).squeeze(-1).squeeze(-1)
        return alignment_mat

#%% load text file
# df = pd.read_csv('test_fx_sentment_v3.csv') # for single currency
df = pd.read_csv('fx_sentment_news_all.csv') # for currency pairs
# df = df[(df['entityid'] == 'USD/JPY') |(df['entityid'] == 'EUR/USD')  | (df['entityid'] == 'GBP/USD') ]
# df = df[df.notna()['body']]
# df = df[df['confidence'] > 60]
# df['headline'] = df['headline'].fillna(' ')
# df['body'] = df['body'].fillna(' ')
df['text'] = df.apply(lambda x : x['headline'] + ' ' +  x['body'], axis=1)
df.rename(columns = {'score': 'target','entityid':'aspect'}, inplace=True) # get target and aspect
df['target'] = df['target'] + 1  # target: 0,1,2
df['target'] = df['target'].astype(int) # from float to int
df = df[['text','target','aspect']]

df['text'] = df['text'].apply(lambda x : x.lower())
df['aspect'] = df['aspect'].apply(lambda x : x.lower())
# df = df[df.notna()['aspect']]
df.shape

#%% language databunch
bs = 32
data_lm = (TextList.from_df(df, cols='text')
           .split_by_rand_pct(0.1)
           .label_for_lm()
           .databunch(bs=bs))
data_lm.save('data_lm_lowercase.pkl')
data_lm = load_data( r'/home/ubuntu/projects/ABSA-PyTorch/', 'data_lm_lowercase.pkl', bs=bs)
learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3) # load pre-trained model

learn_lm.fit_one_cycle(1, 1e-2, moms=(0.8,0.7)) # fine tune the last layer

learn_lm.unfreeze()
learn_lm.fit_one_cycle(10, 1e-3, moms=(0.8,0.7)) # fine tune ALL the layers
learn_lm.save('fine_tuned_lowercase')
learn_lm.load('fine_tuned_lowercase')

#%% test language model performance
TEXT = "ocbc  "
N_WORDS = 30
N_SENTENCES = 2
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

TEXT = "usd "
N_WORDS = 30
N_SENTENCES = 2
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

encoder_lm = learn_lm.model[0] # get encoder for classification model
layers = list(learn_lm.model.children())
embedding_weights = layers[0].encoder.weight  # save embedding weights
embedding_dict = data_lm.vocab.stoi  # save embedding dictionary
learn_lm.save_encoder('fine_tuned_enc_uppercase')  # save encoder



#%% classification pre-processing
#%% split train and validation dataset
data = df.copy()
df = df[(df['aspect'] == 'USD')|(df['aspect'] == 'EUR')|(df['aspect'] == 'JPY')|(df['aspect'] == 'GBP')]

df['text'] = df['text'].apply(lambda x : x.lower())
df['aspect'] = df['aspect'].apply(lambda x : x.lower())
df['aspect'] = df['aspect'].apply(lambda x: x.replace('/',' '))
val_perc = 0.2
df = df.sample(frac=1) # shuffle df rows
df = pd.concat([df[df.target==1][:15000],df[df.target==0][:15000],df[df.target==2][:15000]])  # for unbalanced dataset => [1000, 6000, 3000]
df = df.sample(frac=1) # shuffle df rows


df_val = df[:round(df.shape[0]*val_perc)]
df_train = df[round(df.shape[0]*val_perc):]
df1 = df.copy()



tokenizer = Tokenizer()  # tokenize text (fastai default tokenizer)
df_train['text'] = tokenizer.process_all(df_train['text'])
df_train['aspect'] = tokenizer.process_all(df_train['aspect'])
df_train['text'] = df_train['text'].apply(lambda x: ' '.join(x))
df_train['aspect'] = df_train['aspect'].apply(lambda x: ' '.join(x))

df_val['text'] = tokenizer.process_all(df_val['text'])
df_val['aspect'] = tokenizer.process_all(df_val['aspect'])
df_val['text'] = df_val['text'].apply(lambda x: ' '.join(x))
df_val['aspect'] = df_val['aspect'].apply(lambda x: ' '.join(x))



#%% get text embedding
def label_sentence(s):
    return np.array([embedding_dict.get(x,0) for x in s.split()])

df_train['text'] = df_train['text'].apply(label_sentence)
df_train['aspect'] = df_train['aspect'].apply(label_sentence)

df_val['text'] = df_val['text'].apply(label_sentence)
df_val['aspect'] = df_val['aspect'].apply(label_sentence)

#%% pad text to max_len
max_len = 1000
def pad_and_truncate(sequence, maxlen, dtype='int64'):
    x = (np.ones(maxlen) * 1).astype(dtype) # fastai default padding token is 1
    trunc = sequence[:maxlen]
    # trunc = np.asarray(trunc, dtype=dtype)
    x[:len(trunc)] = trunc
    return x

df_train['text'] = df_train['text'].apply(pad_and_truncate,args=(max_len,))
df_train['aspect'] = df_train['aspect'].apply(pad_and_truncate,args=(max_len,))

df_val['text'] = df_val['text'].apply(pad_and_truncate,args=(max_len,))
df_val['aspect'] = df_val['aspect'].apply(pad_and_truncate,args=(max_len,))

#%% convert into tensor
x_train_torch = torch.tensor(np.stack((df_train['text'].tolist(), df_train['aspect'].tolist()), axis=1), dtype=torch.long)
y_train_torch = torch.tensor(df_train['target'].values, dtype=torch.long)
x_val_torch = torch.tensor(np.stack((df_val['text'].tolist(), df_val['aspect'].tolist()), axis=1), dtype=torch.long)
y_val_torch = torch.tensor(df_val['target'].values, dtype=torch.long)

#%% convert to Pytorch DataLoader
train_dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle=True, drop_last=True)
val_dataset = torch.utils.data.TensorDataset(x_val_torch, y_val_torch)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = bs, shuffle=True, drop_last=True)

#%% convert to fastai DataBunch
databunch = DataBunch(train_dl = train_loader, valid_dl = val_loader)

#%% classifier
# hyper parameters
_model_meta = {AWD_LSTM: {'hid_name':'emb_sz', 'url':URLs.WT103_FWD, 'url_bwd':URLs.WT103_BWD,
                          'config_lm':awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,
                          'config_clas':awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split}
               }
meta = _model_meta[AWD_LSTM]
config = meta['config_clas'].copy()
bptt:int = 70
max_len = max_len
vocab_sz = embedding_weights.shape[0]
drop_mult:float=0.5
lin_ftrs = [50]
ps = [0.1] * len(lin_ftrs)
n_class = 3
layers = [config[meta['hid_name']] * 3] + lin_ftrs + [n_class]
ps = [config.pop('output_p')] + ps
init = config.pop('init') if 'init' in config else None
for k in config.keys():
    if k.endswith('_p'): config[k] *= drop_mult

pad_idx = 1
emb_sz = 400



def pad_tensor(t, bs, val=0.):
    if t.size(0) < bs:
        return torch.cat([t, val + t.new_zeros(bs-t.size(0), *t.shape[1:])])
    return t

# Edited AWD_LSTM
class AWD_LSTM_clas(nn.Module):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182."
    initrange=0.1
    def __init__(self, vocab_sz, emb_sz, n_hid, n_layers, pad_token=1,max_len=max_len,
                 hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5, qrnn:bool=False, bidir:bool=False):
        super().__init__()
        self.bs, self.qrnn, self.emb_sz, self.n_hid, self.n_layers = 1, qrnn, emb_sz, n_hid, n_layers
        self.n_dir = 2 if bidir else 1
        self.max_len = max_len
        self.pad_token = pad_token
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        if self.qrnn:
            #Using QRNN requires an installation of cuda
            from .qrnn import QRNN
            self.rnns = [QRNN(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.n_dir, 1,
                              save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True, bidirectional=bidir)
                         for l in range(n_layers)]
            for rnn in self.rnns:
                rnn.layers[0].linear = WeightDropout(rnn.layers[0].linear, weight_p, layer_names=['weight'])
        else:
            self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.n_dir, 1,
                                 batch_first=True, bidirectional=bidir) for l in range(n_layers)]
            self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input):
        input, aspect = input[0], input[1] # [input text] and [input aspect]
        bs, sl = input.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()
        # mask = (input == self.pad_token)
        # get batch lengths
        # lengths = sl - mask.long().sum(1)
        aspect_len = torch.tensor(torch.sum(aspect != self.pad_token, dim=-1), dtype=torch.float).cuda()
        aspect = self.encoder(aspect)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, self.max_len, -1)  # torch.expand: -1 means not changing the size of that dimension
        # n_empty = (lengths == 0).sum()
        # if n_empty > 0:
        #     input = input[:-n_empty]
        #     lengths = lengths[:-n_empty]
        #     self.hidden = [(h[0][:,:input.size(0)], h[1][:,:input.size(0)]) for h in self.hidden]
        raw_output = self.input_dp(self.encoder_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden, cpu=False)
        return raw_outputs, outputs, aspect

    def _one_hidden(self, l):
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz
        return next(self.parameters()).new(1, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]
class MultiBatchEncoder(Module):
    "Create an encoder over `module` that can process a full sentence."
    def __init__(self, bptt:int, max_len:int, module:nn.Module, pad_idx:int=1):
        self.max_len,self.bptt,self.module,self.pad_idx = max_len,bptt,module,pad_idx
    def concat(self, arrs:Collection[Tensor])->Tensor:
        "Concatenate the `arrs` along the batch dimension."
        return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]
    def reset(self):
        if hasattr(self.module, 'reset'): self.module.reset()
    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:
        # bs, sl = input[:, 0, :].size()
        lengths = torch.tensor(torch.sum(input[:, 0, :] != 1, dim=-1), dtype=torch.float).cuda()
        sl = int(lengths.max())
        context = input[:, 0, :][:, :sl]
        aspect = input[:, 1, :]
        self.reset()
        raw_outputs,outputs,masks = [],[],[]
        for i in range(0, sl, self.bptt):
            r, o , aspect= self.module([context[:, i: min(i + self.bptt, sl)], aspect])
            masks.append(input[:, 0,:][:,i: min(i+self.bptt, sl)] == self.pad_idx)
            raw_outputs.append(r)
            outputs.append(o)
        return self.concat(raw_outputs),self.concat(outputs),torch.cat(masks,dim=1), aspect
class AttentionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = SoftAttention(400, 400).cuda()
        self.dense1 = nn.Linear(400*3, 50).cuda()
        self.dense2 = nn.Linear(50, 3).cuda()
        self.bn0 = nn.BatchNorm1d(400*3).cuda()
        self.bn1 = nn.BatchNorm1d(50).cuda()
        self.fc_dp1 = nn.Dropout(0.2)  # fc dropout
        self.fc_dp2 = nn.Dropout(0.1)  # fc dropout
        self.dense = nn.Linear(400, 3).cuda()

    def forward(self, input):
        raw_outputs,outputs,mask, aspect = input
        output = outputs[-1] # get last layer outputs: [bs, seq_len, em_dim]
        attention_output, hs = self.attention(output, aspect)
        # avg_pool = output.masked_fill(mask[:, :, None], 0).mean(dim=1)
        # avg_pool *= output.size(1) / (output.size(1) - mask.type(avg_pool.dtype).sum(dim=1))[:, None]
        # max_pool = output.masked_fill(mask[:, :, None], -float('inf')).max(dim=1)[0]
        # x = torch.cat([output[:, -1], attention_output, max_pool], 1)
        # x = self.fc_dp1(self.bn0(x))
        # fc1_out = F.relu(self.dense1(x))
        # fc1_out = self.fc_dp2(self.bn1(fc1_out))
        # fc_out = self.dense2(fc1_out)
        out = self.dense(hs)
        return out.cuda()


# IAN
class AWD_LSTM_IAN(nn.Module):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182."
    initrange=0.1
    def __init__(self, vocab_sz, emb_sz, n_hid, n_layers, pad_token=1,max_len=max_len,
                 hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5, qrnn:bool=False, bidir:bool=False):
        super().__init__()
        self.bs, self.qrnn, self.emb_sz, self.n_hid, self.n_layers = 1, qrnn, emb_sz, n_hid, n_layers
        self.n_dir = 2 if bidir else 1
        self.max_len = max_len
        self.pad_token = pad_token
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        if self.qrnn:
            #Using QRNN requires an installation of cuda
            from .qrnn import QRNN
            self.rnns = [QRNN(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.n_dir, 1,
                              save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True, bidirectional=bidir)
                         for l in range(n_layers)]
            for rnn in self.rnns:
                rnn.layers[0].linear = WeightDropout(rnn.layers[0].linear, weight_p, layer_names=['weight'])
        else:
            self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.n_dir, 1,
                                 batch_first=True, bidirectional=bidir) for l in range(n_layers)]
            self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input):
        input, aspect = input[0], input[1] # [input text] and [input aspect]
        bs, sl = input.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()
        raw_output = self.input_dp(self.encoder_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden, cpu=False)
        return raw_outputs, outputs, aspect

    def _one_hidden(self, l):
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz
        return next(self.parameters()).new(1, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]
class MultiBatchEncoder_IAN(Module):
    "Create an encoder over `module` that can process a full sentence."
    def __init__(self, bptt:int, max_len:int, module:nn.Module, pad_idx:int=1):
        self.max_len,self.bptt,self.module,self.pad_idx = max_len,bptt,module,pad_idx
    def concat(self, arrs:Collection[Tensor])->Tensor:
        "Concatenate the `arrs` along the batch dimension."
        return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]
    def reset(self):
        if hasattr(self.module, 'reset'): self.module.reset()
    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:
        bs, sl = input[:, 0, :].size()
        self.reset()
        text = input[:, 0, :]
        text_raw_len = torch.sum(text != 1, dim=-1)
        raw_outputs,outputs,masks = [],[],[]

        for i in range(0, sl, self.bptt):
            r, o , aspect = self.module([input[:, 0,:][:, i: min(i + self.bptt, sl)], input[:, 1,:]])
            masks.append(input[:, 0,:][:,i: min(i+self.bptt, sl)] == self.pad_idx)
            raw_outputs.append(r)
            outputs.append(o)
        return self.concat(raw_outputs),self.concat(outputs),torch.cat(masks,dim=1), aspect, text_raw_len
class AttentionDecoder_IAN(nn.Module):
    def __init__(self, vocab_sz = vocab_sz, emb_sz = emb_sz, pad_token=1, embed_p=0.1):
        super().__init__()
        self.vocab_sz, self.emb_sz, self.pad_token, self.embed_p = vocab_sz, emb_sz, pad_token, embed_p
        self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.emb_dp = EmbeddingDropout(self.emb, embed_p)
        self.lstm_aspect = DynamicLSTM(400, 400, num_layers=1, batch_first=True)
        self.attention_aspect = Attention(400, score_function='bi_linear')
        self.attention_context = Attention(400, score_function='bi_linear')
        # self.bn0 = nn.BatchNorm1d(400 * 2)
        # self.fc_dp0 = nn.Dropout(0.2)  # fc dropout
        # self.dense1 = nn.Linear(400 * 2, 50)
        # self.bn1 = nn.BatchNorm1d(50)
        # self.fc_dp1 = nn.Dropout(0.1)  # fc dropout
        # self.dense2 = nn.Linear(50, 3)
        self.dense = nn.Linear(800, 3)

    def forward(self, input):
        raw_outputs,outputs,mask, aspect, text_raw_len = input

        context = outputs[-1]  # get last layer outputs: [bs, seq_len, em_dim]
        context_pool = torch.sum(context, dim=1)
        text_raw_len = text_raw_len.type(torch.FloatTensor).cuda()
        context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_len = torch.sum(aspect != self.pad_token, dim=-1)
        aspect = self.emb_dp(aspect)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)
        aspect_len = torch.tensor(aspect_len, dtype=torch.float).cuda()

        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = context_final.squeeze(dim=1)
        x = torch.cat((aspect_final, context_final), dim=-1)
        # x = self.fc_dp0(self.bn0(x))
        # fc1_out = F.relu(self.dense1(x))
        # fc1_out = self.fc_dp1(self.bn1(fc1_out))
        # out = self.dense2(fc1_out)
        out = self.dense(x)
        return out.cuda()

# AOA
class AWD_LSTM_AOA(nn.Module):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182."
    initrange=0.1
    def __init__(self, vocab_sz, emb_sz, n_hid, n_layers, pad_token=1,max_len=max_len,
                 hidden_p=config['hidden_p'], input_p=config['input_p'], embed_p=config['embed_p'], weight_p=config['weight_p'], qrnn:bool=False, bidir:bool=False):
        super().__init__()
        self.bs, self.qrnn, self.emb_sz, self.n_hid, self.n_layers = 1, qrnn, emb_sz, n_hid, n_layers
        self.n_dir = 2 if bidir else 1
        self.max_len = max_len
        self.pad_token = pad_token
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        if self.qrnn:
            #Using QRNN requires an installation of cuda
            from .qrnn import QRNN
            self.rnns = [QRNN(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.n_dir, 1,
                              save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True, bidirectional=bidir)
                         for l in range(n_layers)]
            for rnn in self.rnns:
                rnn.layers[0].linear = WeightDropout(rnn.layers[0].linear, weight_p, layer_names=['weight'])
        else:
            self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.n_dir, 1,
                                 batch_first=True, bidirectional=bidir) for l in range(n_layers)]
            self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input):
        input, aspect = input[0], input[1] # [input text] and [input aspect]
        bs, sl = input.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()
        raw_output = self.input_dp(self.encoder_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden, cpu=False)
        return raw_outputs, outputs, aspect

    def _one_hidden(self, l):
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz
        return next(self.parameters()).new(1, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]
class MultiBatchEncoder_AOA(Module):
    "Create an encoder over `module` that can process a full sentence."
    def __init__(self, bptt:int, max_len:int, module:nn.Module, pad_idx:int=config['pad_token']):
        self.max_len,self.bptt,self.module,self.pad_idx = max_len,bptt,module,pad_idx
    def concat(self, arrs:Collection[Tensor])->Tensor:
        "Concatenate the `arrs` along the batch dimension."
        return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]
    def reset(self):
        if hasattr(self.module, 'reset'): self.module.reset()
    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:

        lengths = torch.tensor(torch.sum(input[:, 0, :] != 1, dim=-1), dtype=torch.float).cuda()
        sl = int(lengths.max())
        context = input[:, 0, :][:, :sl]
        text_raw_len = torch.sum(context != 1, dim=-1)
        aspect = input[:, 1, :]
        self.reset()
        raw_outputs,outputs,masks = [],[],[]
        for i in range(0, sl, self.bptt):
            r, o , aspect= self.module([context[:, i: min(i + self.bptt, sl)], aspect])
            masks.append(input[:, 0,:][:,i: min(i+self.bptt, sl)] == self.pad_idx)
            raw_outputs.append(r)
            outputs.append(o)
        return self.concat(raw_outputs),self.concat(outputs),torch.cat(masks,dim=1), aspect, text_raw_len
class AttentionDecoder_AOA(nn.Module):
    def __init__(self, vocab_sz = vocab_sz, emb_sz = emb_sz, pad_token=1, embed_p=0.1):
        super().__init__()
        self.vocab_sz, self.emb_sz, self.pad_token, self.embed_p = vocab_sz, emb_sz, pad_token, embed_p
        # self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.emb = nn.Embedding.from_pretrained(embedding_weights)
        # self.emb_dp = EmbeddingDropout(self.emb, embed_p)
        self.asp_lstm = DynamicLSTM(400, 400, num_layers = 1, batch_first=True, bidirectional=False)
        # self.dense1 = nn.Linear(400*3, 50)
        # self.dense2 = nn.Linear(50, 3)
        # self.fc_dp0 = nn.Dropout(0.2)  # fc dropout
        # self.fc_dp2 = nn.Dropout(0.1)  # fc dropout
        # self.bn0 = nn.BatchNorm1d(400*3)
        # self.bn1 = nn.BatchNorm1d(50)
        # self.bn2 = nn.BatchNorm1d(50)
        self.dense = nn.Linear(400, 3)


    def forward(self, input):
        raw_outputs,outputs,mask, aspect, text_raw_len = input
        asp_len  = torch.sum(aspect != self.pad_token, dim=-1)
        asp = self.emb(aspect)

        ctx_out = outputs[-1]  # get last layer outputs: [bs, seq_len, em_dim]
        # asp_out = asp
        asp_out, (_, _) = self.asp_lstm(asp, asp_len)  # batch_size x (asp) seq_len x hidden_dim
        interaction_mat = torch.matmul(ctx_out,
                                       torch.transpose(asp_out, 1, 2))  # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = F.softmax(interaction_mat, dim=1)  # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = F.softmax(interaction_mat, dim=2)  # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(dim=1, keepdim=True)  # batch_size x 1 x (asp) seq_len
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2))  # batch_size x (ctx) seq_len x 1
        weighted_sum = torch.matmul(torch.transpose(ctx_out, 1, 2), gamma).squeeze(-1)  # batch_size x hidden_dim

        # max_pool = ctx_out.masked_fill(mask[:, :, None], -float('inf')).max(dim=1)[0]
        # avg_pool = ctx_out.masked_fill(mask[:, :, None], 0).mean(dim=1)
        # avg_pool *= ctx_out.size(1) / (ctx_out.size(1) - mask.type(avg_pool.dtype).sum(dim=1))[:, None]

        # x = torch.cat([ctx_out[:, -1], weighted_sum, max_pool], 1)
        # x = self.fc_dp0(self.bn0(x))
        # fc1_out = F.relu(self.dense1(x))
        # fc1_out = self.fc_dp2(self.bn1(fc1_out))
        # fc_out = self.dense2(fc1_out)

        # fc1_out = F.relu(self.dense1(weighted_sum)) # batch_size x polarity_dim
        # fc1_out = self.fc_dp2(self.bn1(fc1_out))
        fc_out = self.dense(weighted_sum)
        return fc_out.cuda()

# MGAN
class AWD_LSTM_MGAN(nn.Module):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182."
    initrange=0.1
    def __init__(self, vocab_sz, emb_sz, n_hid, n_layers, pad_token=1,max_len=max_len,
                 hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5, qrnn:bool=False, bidir:bool=False):
        super().__init__()
        self.bs, self.qrnn, self.emb_sz, self.n_hid, self.n_layers = 1, qrnn, emb_sz, n_hid, n_layers
        self.n_dir = 2 if bidir else 1
        self.max_len = max_len
        self.pad_token = pad_token
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        if self.qrnn:
            #Using QRNN requires an installation of cuda
            from .qrnn import QRNN
            self.rnns = [QRNN(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.n_dir, 1,
                              save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True, bidirectional=bidir)
                         for l in range(n_layers)]
            for rnn in self.rnns:
                rnn.layers[0].linear = WeightDropout(rnn.layers[0].linear, weight_p, layer_names=['weight'])
        else:
            self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.n_dir, 1,
                                 batch_first=True, bidirectional=bidir) for l in range(n_layers)]
            self.rnns = [WeightDropout(rnn, weight_p) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input):
        input, aspect = input[0], input[1] # [input text] and [input aspect]
        bs, sl = input.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()
        raw_output = self.input_dp(self.encoder_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden, cpu=False)
        return raw_outputs, outputs, aspect

    def _one_hidden(self, l):
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz
        return next(self.parameters()).new(1, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]
class MultiBatchEncoder_MGAN(Module):
    "Create an encoder over `module` that can process a full sentence."
    def __init__(self, bptt:int, max_len:int, module:nn.Module, pad_idx:int=1):
        self.max_len,self.bptt,self.module,self.pad_idx = max_len,bptt,module,pad_idx
    def concat(self, arrs:Collection[Tensor])->Tensor:
        "Concatenate the `arrs` along the batch dimension."
        return [torch.cat([l[si] for l in arrs], dim=1) for si in range_of(arrs[0])]
    def reset(self):
        if hasattr(self.module, 'reset'): self.module.reset()
    def forward(self, input:LongTensor)->Tuple[Tensor,Tensor]:
        # bs, sl = input[:, 0, :].size()
        lengths = torch.tensor(torch.sum(input[:, 0, :] != 1, dim=-1), dtype=torch.float).cuda()
        sl = int(lengths.max())
        context = input[:, 0, :][:, :sl]
        text_raw_len = torch.sum(context != 1, dim=-1)
        aspect = input[:, 1, :]
        self.reset()
        raw_outputs,outputs,masks = [],[],[]
        for i in range(0, sl, self.bptt):
            r, o , aspect= self.module([context[:, i: min(i + self.bptt, sl)], aspect])
            masks.append(input[:, 0,:][:,i: min(i+self.bptt, sl)] == self.pad_idx)
            raw_outputs.append(r)
            outputs.append(o)
        return self.concat(raw_outputs),self.concat(outputs),torch.cat(masks,dim=1), aspect, text_raw_len
class AttentionDecoder_MGAN(nn.Module):
    def __init__(self, vocab_sz = vocab_sz, emb_sz = emb_sz, pad_token=1, embed_p=0.1, batch_size = bs):
        super().__init__()
        self.batch_size = batch_size
        self.vocab_sz, self.emb_sz, self.pad_token, self.embed_p = vocab_sz, emb_sz, pad_token, embed_p
        self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.emb_dp = EmbeddingDropout(self.emb, embed_p)
        self.asp_lstm = DynamicLSTM(400, 400, num_layers = 1, batch_first=True, bidirectional=False)

        self.w_a2c = nn.Parameter(torch.Tensor(400, 400))
        self.w_c2a = nn.Parameter(torch.Tensor(400, 400))
        self.alignment = AlignmentMatrix()

        self.dense = nn.Linear(400*4, 3)
        self.dense1 = nn.Linear(400, 50)
        self.dense2 = nn.Linear(50, 3)
        self.fc_dp1 = nn.Dropout(0.2)  # fc dropout
        self.fc_dp2 = nn.Dropout(0.1)  # fc dropout
        self.bn1 = nn.BatchNorm1d(400)
        self.bn2 = nn.BatchNorm1d(50)

    def forward(self, input):

        raw_outputs, outputs, mask, aspect, text_raw_len = input
        # batch_size = aspect.shape[0]
        asp_len  = torch.sum(aspect != self.pad_token, dim=-1)
        asp = self.emb_dp(aspect)

        ctx_out = outputs[-1]  # get last layer outputs: [bs, seq_len, em_dim]
        asp_out, (_, _) = self.asp_lstm(asp, asp_len)  # batch_size x (asp) seq_len x hidden_dim

        ctx_pool = torch.sum(ctx_out, dim=1)
        ctx_pool = torch.div(ctx_pool, text_raw_len.float().unsqueeze(-1)).unsqueeze(-1)  # batch_size x hidden_dim x 1

        asp_pool = torch.sum(asp_out, dim=1)
        asp_pool = torch.div(asp_pool, asp_len.float().unsqueeze(-1)).unsqueeze(-1)  # batch_size x hidden_dim x 1

        alignment_mat = self.alignment(self.batch_size, ctx_out, asp_out)  # batch_size x (ctx)seq_len x (asp)seq_len

        f_asp2ctx = torch.matmul(ctx_out.transpose(1, 2),
                                 F.softmax(alignment_mat.max(2, keepdim=True)[0], dim=1)).squeeze(-1)
        f_ctx2asp = torch.matmul(F.softmax(alignment_mat.max(1, keepdim=True)[0], dim=2), asp_out).transpose(1, 2).squeeze(-1)

        c_asp2ctx_alpha = F.softmax(ctx_out.matmul(self.w_a2c.expand(self.batch_size, -1, -1)).matmul(asp_pool), dim=1)
        c_asp2ctx = torch.matmul(ctx_out.transpose(1, 2), c_asp2ctx_alpha).squeeze(-1)
        c_ctx2asp_alpha = F.softmax(asp_out.matmul(self.w_c2a.expand(self.batch_size, -1, -1)).matmul(ctx_pool), dim=1)
        c_ctx2asp = torch.matmul(asp_out.transpose(1, 2), c_ctx2asp_alpha).squeeze(-1)

        feat = torch.cat([c_asp2ctx, f_asp2ctx, f_ctx2asp, c_ctx2asp], dim=1)
        out = self.dense(feat)  # bathc_size x polarity_dim

        # # out = self.dense(weighted_sum)  # batch_size x polarity_dim
        # x = self.fc_dp1(self.bn1(weighted_sum))
        # fc1_out = F.relu(self.dense1(x))
        # fc1_out = self.fc_dp2(self.bn2(fc1_out))
        # out = self.dense2(fc1_out)
        return out.cuda()


#%% AWD_LSTM_Attention
module = AWD_LSTM_clas(vocab_sz, **config)
model = SequentialRNN(MultiBatchEncoder(bptt, max_len, module, pad_idx=pad_idx), AttentionDecoder())
learn = Learner(databunch, model, loss_func = F.cross_entropy, metrics=accuracy)
pretrained_dict = encoder_lm.state_dict()
model_dict = learn.model[0].state_dict()
pretrained_encoder = {key: value for (key, value) in zip(model_dict.keys(), pretrained_dict.values())}
learn.model[0].load_state_dict(pretrained_encoder)
# cut the model with different layer groups
learn.split([learn.model[0].module.rnns[:1],learn.model[0].module.rnns[1], learn.model[0].module.rnns[2:], learn.model[1]])

# tune the last layer
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
# tune last 2 layers
learn.freeze_to(-2)
learn.fit_one_cycle(1,  slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
# tune last 3 layers
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3),moms=(0.8,0.7))
# tune all layers
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

learn.save('fine_ture_classification_ataelstm')
#%% error analysis
preds, y,losses = learn.get_preds(with_loss=True)
confusion_matrix(y, preds.max(1)[1])


#%% Interactive Attention Networks
bptt = 70
module = AWD_LSTM_IAN(vocab_sz, **config)
model = SequentialRNN(MultiBatchEncoder_IAN(bptt, max_len, module, pad_idx=pad_idx), AttentionDecoder_IAN())
learn = Learner(databunch, model, loss_func = F.cross_entropy, metrics=accuracy)
pretrained_dict = encoder_lm.state_dict()
model_dict = learn.model[0].state_dict()
pretrained_encoder = {key: value for (key, value) in zip(model_dict.keys(), pretrained_dict.values())}
learn.model[0].load_state_dict(pretrained_encoder)
# cut the model with different layer groups
learn.split([learn.model[0].module.rnns[:1],learn.model[0].module.rnns[1], learn.model[0].module.rnns[2:], learn.model[1]])

# tune the last layer
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
# tune last 2 layers
learn.freeze_to(-2)
learn.fit_one_cycle(1,  slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
# tune last 3 layers
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3),moms=(0.8,0.7))
# tune all layers
learn.unfreeze()
learn.fit_one_cycle(10, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

def awd_lstm_clas_split(model:nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."
    groups = [[model[0].module.encoder, model[0].module.encoder_dp]]
    groups += [[rnn, dp] for rnn, dp in zip(model[0].module.rnns, model[0].module.hidden_dps)]
    return groups + [[model[1]]]

def split(self, split_on:SplitFuncOrIdxList)->None:
    "Split the model at `split_on`."
    if isinstance(split_on,Callable): split_on = split_on(self.model)
    self.layer_groups = split_model(self.model, split_on)
    return self

#%% AOA
module = AWD_LSTM_AOA(vocab_sz, **config)
model = SequentialRNN(MultiBatchEncoder_AOA(bptt, max_len, module, pad_idx=pad_idx), AttentionDecoder_AOA())
learn_aoa = Learner(databunch, model, loss_func = F.cross_entropy, metrics=accuracy)
pretrained_dict = encoder_lm.state_dict()
model_dict = learn_aoa.model[0].state_dict()
pretrained_encoder = {key: value for (key, value) in zip(model_dict.keys(), pretrained_dict.values())}
learn_aoa.model[0].load_state_dict(pretrained_encoder)
# cut the model with different layer groups
split_on = awd_lstm_clas_split(learn_aoa.model)
learn_aoa.split(split_on)


# tune the last layer
learn_aoa.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))# tune last 2 layers
learn_aoa.freeze_to(-2)
learn_aoa.fit_one_cycle(1,  slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
# tune last 3 layers
learn_aoa.freeze_to(-3)
learn_aoa.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3),moms=(0.8,0.7))

# tune all layers
learn_aoa.unfreeze()
learn_aoa.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

learn_aoa.save('fine_tune_classification_pairs_aoa')


#%% MGAN
module = AWD_LSTM_MGAN(vocab_sz, **config)
model = SequentialRNN(MultiBatchEncoder_MGAN(bptt, max_len, module, pad_idx=pad_idx), AttentionDecoder_MGAN())
learn = Learner(databunch, model, loss_func = F.cross_entropy, metrics=accuracy)
pretrained_dict = encoder_lm.state_dict()
model_dict = learn.model[0].state_dict()
pretrained_encoder = {key: value for (key, value) in zip(model_dict.keys(), pretrained_dict.values())}
learn.model[0].load_state_dict(pretrained_encoder)
# cut the model with different layer groups
learn.split([learn_aoa.model[0].module.encoder_dp, learn.model[0].module.rnns[0],learn.model[0].module.rnns[1], learn.model[0].module.rnns[2], learn.model[1]])

# tune the last layer
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))# tune last 2 layers
learn.freeze_to(-2)
learn.fit_one_cycle(1,  slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))

# tune last 3 layers
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3),moms=(0.8,0.7))

# tune all layers
learn.unfreeze()
learn.fit_one_cycle(10, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

learn.save('fine_ture_classification_pairs_aoa')



#%% document level classifier
bs = 32
data_clas = (TextList.from_df(df1, cols='text',vocab=data_lm.vocab)
             .split_by_rand_pct(0.2)
             .label_from_df(cols='target')
             .databunch(bs=bs))
data_clas.save('data_clas.pkl')
data_clas = load_data( r'/home/ubuntu/projects/ABSA-PyTorch/', 'data_clas.pkl', bs=bs)
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc_currency_pair_lowercase')
# fine-tune the last layer
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
# fine tune the last 2 layers
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
learn.save('doc_level_classifier_fastai')

