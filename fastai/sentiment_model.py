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

import warnings
warnings.filterwarnings("ignore")

class AWD_LSTM1(nn.Module):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182."
    initrange=0.1

    def __init__(self, vocab_sz, emb_sz, n_hid, n_layers, pad_token,
                 hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5):
        super().__init__()
        self.bs,self.emb_sz,self.n_hid,self.n_layers,self.pad_token = 1,emb_sz,n_hid,n_layers,pad_token
        self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.emb_dp = EmbeddingDropout(self.emb, embed_p)
        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz), 1,
                             batch_first=True) for l in range(n_layers)]
        self.rnns = nn.ModuleList([WeightDropout(rnn, weight_p) for rnn in self.rnns])
        self.emb.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input):
        bs,sl = input.size()
        mask = (input == self.pad_token)
        lengths = sl - mask.long().sum(1)
        n_empty = (lengths == 0).sum()
        if n_empty > 0:
            input = input[:-n_empty]
            lengths = lengths[:-n_empty]
            self.hidden = [(h[0][:,:input.size(0)], h[1][:,:input.size(0)]) for h in self.hidden]
        raw_output = self.input_dp(self.emb_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output = pack_padded_sequence(raw_output, lengths, batch_first=True)
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            raw_output = pad_packed_sequence(raw_output, batch_first=True)[0]
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
            new_hidden.append(new_h)
        self.hidden = to_detach(new_hidden)
        return raw_outputs, outputs, mask

    def _one_hidden(self, l):
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz
        return next(self.parameters()).new(1, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]

class AWD_LSTM(Module):
    "AWD-LSTM/QRNN inspired by https://arxiv.org/abs/1708.02182."

    initrange=0.1

    def __init__(self, vocab_sz:int, emb_sz:int, n_hid:int, n_layers:int, pad_token:int=1, hidden_p:float=0.2,
                 input_p:float=0.6, embed_p:float=0.1, weight_p:float=0.5, qrnn:bool=False, bidir:bool=False):
        self.bs,self.qrnn,self.emb_sz,self.n_hid,self.n_layers = 1,qrnn,emb_sz,n_hid,n_layers
        self.n_dir = 2 if bidir else 1
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

    def forward(self, input:Tensor, from_embeddings:bool=False)->Tuple[Tensor,Tensor]:
        if from_embeddings: bs,sl,es = input.size()
        else: bs,sl = input.size()
        if bs!=self.bs:
            self.bs=bs
            self.reset()
        raw_output = self.input_dp(input if from_embeddings else self.encoder_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
        self.hidden = to_detach(new_hidden, cpu=False)
        return raw_outputs, outputs

    def _one_hidden(self, l:int)->Tensor:
        "Return one hidden state."
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
        return one_param(self).new(self.n_dir, self.bs, nh).zero_()

    def select_hidden(self, idxs):
        if self.qrnn: self.hidden = [h[:,idxs,:] for h in self.hidden]
        else: self.hidden = [(h[0][:,idxs,:],h[1][:,idxs,:]) for h in self.hidden]
        self.bs = len(idxs)

    def reset(self):
        "Reset the hidden states."
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        if self.qrnn: self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]
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
class NoQueryAttention(Attention):
    '''q is a parameter'''
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)

__all__ = ['RNNLearner', 'LanguageLearner', 'convert_weights', 'decode_spec_tokens', 'get_language_model', 'language_model_learner',
           'MultiBatchEncoder', 'get_text_classifier', 'text_classifier_learner', 'PoolingLinearClassifier']

_model_meta = {AWD_LSTM: {'hid_name':'emb_sz', 'url':URLs.WT103_FWD, 'url_bwd':URLs.WT103_BWD,
                          'config_lm':awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,
                          'config_clas':awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split},
               Transformer: {'hid_name':'d_model', 'url':URLs.OPENAI_TRANSFORMER,
                             'config_lm':tfmer_lm_config, 'split_lm': tfmer_lm_split,
                             'config_clas':tfmer_clas_config, 'split_clas': tfmer_clas_split},
               TransformerXL: {'hid_name':'d_model',
                              'config_lm':tfmerXL_lm_config, 'split_lm': tfmerXL_lm_split,
                              'config_clas':tfmerXL_clas_config, 'split_clas': tfmerXL_clas_split}}

# open a file
def load_text_data(file):
    fin = open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    text_raw_list=[]
    aspect_list=[]
    target_list=[]
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        polarity = lines[i + 2].strip()
        text = text_left + " " + aspect + " " + text_right
        text_raw_list.append(text)
        aspect_list.append(aspect)
        target_list.append(polarity)
    df = pd.DataFrame(data = {'text': text_raw_list,'aspect': aspect_list,'target': target_list})
    df['target'] = df['target'].apply(lambda x: int(x)+1)
    return df
train_file = r'C:\Users\xiaoxu\Documents\ABSA-PyTorch\datasets\semeval14\Laptops_Train.xml.seg'
test_file = r'C:\Users\xiaoxu\Documents\ABSA-PyTorch\datasets\semeval14\Laptops_Test_Gold.xml.seg'
train_df =  load_text_data(train_file)
test_df =  load_text_data(test_file)
df = pd.concat([train_df, test_df])


# language databunch
bs = 64
data_lm = (TextList.from_df(df, cols='text')
           .split_by_rand_pct(0.1)
           .label_for_lm()
           .databunch(bs=bs))
data_lm.save('data_lm.pkl')
# load pre-trained model
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
# fine tune the last layer
learn.fit_one_cycle(1, 1e-2, moms=(0.8,0.7))
learn.save('fit_head')
# fine tune ALL the layers
learn.unfreeze()
learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
learn.save('fine_tuned')

# view model structure
m = learn.model
# m.eval()
encoder = learn.model
# save embedding matrix
layers = list(m.children())
embedding_weights = layers[0].encoder.weight
embedding_dict = data_lm.vocab.stoi
# save encoder
learn.save_encoder('fine_tuned_enc')
#
# # create classification databunch
# bs = 64
# data_clas = (TextList.from_df(df, cols='text',vocab=data_lm.vocab)
#              .split_from_df()
#              .label_from_df(cols='target')
#              .databunch(bs=bs))
# data_clas.save('data_clas.pkl')
#
# # load pre-trained sentiment model
# learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
# learn.load_encoder('fine_tuned_enc')
#
# # model components
# encoder = learn.model
# encoder =  awd_lstm[0]
# decoder = awd_lstm[1]


# myLinear = nn.Linear(in_features=4104, out_features=512, bias=True)
# head = nn.Sequential(*list(head.children())[:3], End(), nn.Dropout(.25), myLinear, *list(head.children())[5:])
# model = nn.Sequential(encoder,decoder).cuda()
# learn.model = model

# Load classification data

# convert into torch tensor

## prepare data:
# Input => [text, aspect] ; Output => label
def label_sentence(s):
    return np.array([embedding_dict.get(x,0) for x in s.split()])

df['text'] = df['text'].apply(label_sentence)
df['aspect'] = df['aspect'].apply(label_sentence)

max_seq_len = 70
def pad_and_truncate(sequence, maxlen, dtype='int64'):
    ### fastai default padding token is 1
    x = (np.ones(maxlen) * 1).astype(dtype)
    trunc = sequence[:maxlen]
    # trunc = np.asarray(trunc, dtype=dtype)
    x[:len(trunc)] = trunc
    return x
df1 = df.copy()
df1['text'] = df1['text'].apply(pad_and_truncate,args=(max_seq_len,))
df1['aspect'] = df1['aspect'].apply(pad_and_truncate,args=(max_seq_len,))

# split train and validation dataset
val_perc = 0.2
df_val = df1[:round(len(df)*val_perc)]
df_train = df1[round(len(df)*val_perc):]

# convert into tensor
x_train_torch = torch.tensor(np.stack((df_train['text'].tolist(), df_train['aspect'].tolist()), axis=1), dtype=torch.long)
y_train_torch = torch.tensor(df_train['target'].values, dtype=torch.long)
x_val_torch = torch.tensor(np.stack((df_val['text'].tolist(), df_val['aspect'].tolist()), axis=1), dtype=torch.long)
y_val_torch = torch.tensor(df_val['target'].values, dtype=torch.long)

# convert to Pytorch DataLoader
train_dataset = torch.utils.data.TensorDataset(x_train_torch, y_train_torch)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(x_val_torch, y_val_torch)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = bs, shuffle=True)
# convert to fastai DataBunch
databunch = DataBunch(train_dl = train_loader, valid_dl = val_loader)

# define model structure

# input databunch, model, loss function, metrics

# arch = AWD_LSTM
# meta = _model_meta[arch]
# config = meta['config_clas'].copy()
# lin_ftrs = [50]
# ps = [0.1]*len(lin_ftrs)
# layers = [config[meta['hid_name']] * 3] + lin_ftrs + [n_class]
# n_class = 3
# layers = [config[meta['hid_name']] * 3] + lin_ftrs + [n_class]
# init = config.pop('init') if 'init' in config else None
# bptt:int=70
# max_len:int=20*70
# vocab_sz = len(data_lm.vocab.itos)
# pad_idx: int = 1
# init = config.pop('init') if 'init' in config else None


# bs = 64
# awd_lstm = AWD_LSTM(vocab_sz, **config)
# encoder = MultiBatchEncoder(bptt, max_len, arch(vocab_sz, **config), pad_idx=pad_idx)
# model = SequentialRNN(encoder, PoolingLinearClassifier(layers, ps))
# learn = Learner(databunch, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
# learn.unfreeze()
# learn.fit_one_cycle(1, 2e-3, moms=(0.8,0.7))




# PyTorch puts 0s everywhere we had padding in the output when unpacking.


# Pooling for the classification model:
# - the last hidden state
# - the average of all the hidden states
# - the maximum of all the hidden states

def pad_tensor(t, bs, val=0.):
    if t.size(0) < bs:
        return torch.cat([t, val + t.new_zeros(bs-t.size(0), *t.shape[1:])])
    return t

class AWD_LSTM_clas_1(nn.Module):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182."
    initrange=0.1

    def __init__(self, vocab_sz, emb_sz, n_hid, n_layers, pad_token,qrnn=False,bidir=False,
                 hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5):
        super().__init__()
        self.bs,self.emb_sz,self.n_hid,self.n_layers,self.pad_token = 1,emb_sz,n_hid,n_layers,pad_token
        self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.emb_dp = EmbeddingDropout(self.emb, embed_p)
        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz), 1,
                             batch_first=True) for l in range(n_layers)]
        self.rnns = nn.ModuleList([WeightDropout(rnn, weight_p) for rnn in self.rnns])
        self.emb.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input):
        input, aspect = input[0], input[1] # [input text] and [input aspect]
        bs, sl = input.size()
        mask = (input == self.pad_token)
        # get batch lengths
        lengths = sl - mask.long().sum(1)
        aspect_len = torch.tensor(torch.sum(aspect != self.pad_token, dim=-1), dtype=torch.float).cuda()
        aspect = self.emb(aspect)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, max(lengths), -1)  # torch.expand: -1 means not changing the size of that dimension
        n_empty = (lengths == 0).sum()
        if n_empty > 0:
            input = input[:-n_empty]
            lengths = lengths[:-n_empty]
            self.hidden = [(h[0][:,:input.size(0)], h[1][:,:input.size(0)]) for h in self.hidden]
        raw_output = self.input_dp(self.emb_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output = pack_padded_sequence(raw_output, lengths, batch_first=True, enforce_sorted=False)
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            raw_output = pad_packed_sequence(raw_output, batch_first=True)[0]
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
            new_hidden.append(new_h)
        self.hidden = to_detach(new_hidden)

        return raw_outputs, outputs, mask, aspect

    def _one_hidden(self, l):
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz
        return next(self.parameters()).new(1, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]
class SentenceEncoder(nn.Module):
    def __init__(self, module, bptt, pad_idx=1):
        super().__init__()
        self.bptt, self.module, self.pad_idx = bptt, module, pad_idx
    def concat(self, arrs, bs):
        return [torch.cat([pad_tensor(l[si], bs) for l in arrs], dim=1) for si in range(len(arrs[0]))]
    def forward(self, input):
        bs, sl = input[:, 0,:].size()
        self.module.bs = bs
        self.module.reset()
        raw_outputs, outputs, masks = [], [], []
        for i in range(0, sl, self.bptt):
            r, o, m, aspect = self.module([input[:, 0,:][:, i: min(i + self.bptt, sl)], input[:, 1,:]])
            masks.append(pad_tensor(m, bs, 1))
            raw_outputs.append(r)
            outputs.append(o)
        return self.concat(raw_outputs, bs), self.concat(outputs, bs), torch.cat(masks, dim=1), aspect



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
        bs, sl = input[:, 0, :].size()
        self.reset()
        raw_outputs,outputs,masks = [],[],[]
        for i in range(0, sl, self.bptt):
            r, o , aspect= self.module([input[:, 0,:][:, i: min(i + self.bptt, sl)], input[:, 1,:]])
            masks.append(input[:, 0,:][:,i: min(i+self.bptt, sl)] == self.pad_idx)
            raw_outputs.append(r)
            outputs.append(o)
        return self.concat(raw_outputs),self.concat(outputs),torch.cat(masks,dim=1),aspect
class AttentionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = NoQueryAttention(400 + 400, score_function='bi_linear').cuda()
        self.dense = nn.Linear(400, 3).cuda()

    def forward(self, input):
        raw_outputs,outputs,mask, aspect = input
        output = outputs[-1] # get last layer outputs: [bs, seq_len, em_dim]
        ha = torch.cat((output, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, output), dim=1)
        out = self.dense(output)
        return out.cuda()


_model_meta = {AWD_LSTM: {'hid_name':'emb_sz', 'url':URLs.WT103_FWD, 'url_bwd':URLs.WT103_BWD,
                          'config_lm':awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,
                          'config_clas':awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split},
               Transformer: {'hid_name':'d_model', 'url':URLs.OPENAI_TRANSFORMER,
                             'config_lm':tfmer_lm_config, 'split_lm': tfmer_lm_split,
                             'config_clas':tfmer_clas_config, 'split_clas': tfmer_clas_split},
               TransformerXL: {'hid_name':'d_model',
                              'config_lm':tfmerXL_lm_config, 'split_lm': tfmerXL_lm_split,
                              'config_clas':tfmerXL_clas_config, 'split_clas': tfmerXL_clas_split}}

meta = _model_meta[AWD_LSTM]
config = meta['config_clas'].copy()
bptt:int=70
max_len = 70
vocab_sz = embedding_weights.shape[0]
drop_mult:float=1
lin_ftrs = [50]
ps = [0.1] * len(lin_ftrs)
n_class = 3
layers = [config[meta['hid_name']] * 3] + lin_ftrs + [n_class]
ps = [config.pop('output_p')] + ps
init = config.pop('init') if 'init' in config else None
for k in config.keys():
    if k.endswith('_p'): config[k] *= drop_mult
bptt:int=70
pad_idx = 1


module = AWD_LSTM_clas(vocab_sz, **config)
encoder = MultiBatchEncoder(bptt, max_len, module, pad_idx=pad_idx)
model = SequentialRNN(encoder, AttentionDecoder())
learn = Learner(databunch, model, loss_func = F.cross_entropy, metrics=accuracy)
learn.load_encoder('fine_tuned_enc')


learn.fit_one_cycle(1, 2e-3, moms=(0.8,0.7))
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3, moms=(0.8,0.7))


encoder = SentenceEncoder(AWD_LSTM_clas_1(vocab_sz, **config),bptt)
model = SequentialRNN(encoder, AttentionDecoder())
learn = Learner(databunch, model, loss_func = F.cross_entropy, metrics=accuracy)
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3, moms=(0.8,0.7))