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
m.eval()
encoder = learn.model
# save embedding matrix
layers = list(m.children())
embedding_weights = layers[0].encoder.weight
embedding_dict = data_lm.vocab.stoi
# save encoder
learn.save_encoder('fine_tuned_enc')

# create classification databunch
bs = 64
data_clas = (TextList.from_df(df, cols='text',vocab=data_lm.vocab)
             .split_from_df()
             .label_from_df(cols='target')
             .databunch(bs=bs))
data_clas.save('data_clas.pkl')

# load pre-trained sentiment model
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')

# model components
encoder = learn.model
encoder =  awd_lstm[0]
decoder = awd_lstm[1]


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

max_seq_len = 80
def pad_and_truncate(sequence, maxlen, dtype='int64'):
    x = (np.ones(maxlen) * 0).astype(dtype)
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

arch = AWD_LSTM
meta = _model_meta[arch]
config = meta['config_clas'].copy()
lin_ftrs = [50]
ps = [0.1]*len(lin_ftrs)
layers = [config[meta['hid_name']] * 3] + lin_ftrs + [n_class]
n_class = 3
layers = [config[meta['hid_name']] * 3] + lin_ftrs + [n_class]
init = config.pop('init') if 'init' in config else None
bptt:int=70
max_len:int=20*70
vocab_sz = len(data_lm.vocab.itos)
pad_idx: int = 1
init = config.pop('init') if 'init' in config else None


bs = 64
awd_lstm = AWD_LSTM(vocab_sz, **config)
encoder = MultiBatchEncoder(bptt, max_len, arch(vocab_sz, **config), pad_idx=pad_idx)
model = SequentialRNN(encoder, PoolingLinearClassifier(layers, ps))
learn = Learner(databunch, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
learn.unfreeze()
learn.fit_one_cycle(1, 2e-3, moms=(0.8,0.7))




# PyTorch puts 0s everywhere we had padding in the output when unpacking.
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


# Pooling for the classification model:
# - the last hidden state
# - the average of all the hidden states
# - the maximum of all the hidden states


def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None):
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

class PoolingLinearClassifier(nn.Module):
    "Create a linear classifier with pooling."
    def __init__(self, layers, drops):
        super().__init__()
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers).cuda()

    def forward(self, input):
        raw_outputs,outputs,mask = input
        output = outputs[-1]
        lengths = output.size(1) - mask.long().sum(dim=1)
        avg_pool = output.masked_fill(mask[:,:,None], 0).sum(dim=1)
        avg_pool.div_(lengths.type(avg_pool.dtype)[:,None])
        max_pool = output.masked_fill(mask[:,:,None], -float('inf')).max(dim=1)[0]
        x = torch.cat([output[torch.arange(0, output.size(0)),lengths-1], max_pool, avg_pool], 1) # Concat pooling.
        x = self.layers(x)
        return x.cuda()

def pad_tensor(t, bs, val=0.):
    if t.size(0) < bs:
        return torch.cat([t, val + t.new_zeros(bs-t.size(0), *t.shape[1:])])
    return t


class SentenceEncoder(nn.Module):
    def __init__(self, module, bptt, pad_idx=1):
        super().__init__()
        self.bptt, self.module, self.pad_idx = bptt, module, pad_idx

    def concat(self, arrs, bs):
        return [torch.cat([pad_tensor(l[si], bs) for l in arrs], dim=1) for si in range(len(arrs[0]))]

    def forward(self, input):
        input = input[:, 0,:]
        bs, sl = input.size()
        self.module.bs = bs
        self.module.reset()
        raw_outputs, outputs, masks = [], [], []
        for i in range(0, sl, self.bptt):
            r, o, m = self.module(input[:, i: min(i + self.bptt, sl)])
            masks.append(pad_tensor(m, bs, 1))
            raw_outputs.append(r)
            outputs.append(o)
        return self.concat(raw_outputs, bs), self.concat(outputs, bs), torch.cat(masks, dim=1)


def get_text_classifier(vocab_sz, emb_sz, n_hid, n_layers, n_out, pad_token, bptt, output_p=0.4, hidden_p=0.2,
                        input_p=0.6, embed_p=0.1, weight_p=0.5, layers=None, drops=None):
    "To create a full AWD-LSTM"
    rnn_enc = AWD_LSTM1(vocab_sz, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token,
                        hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    enc = SentenceEncoder(rnn_enc, bptt)
    if layers is None: layers = [50]
    if drops is None:  drops = [0.1] * len(layers)
    layers = [3 * emb_sz] + layers + [n_out]
    drops = [output_p] + drops
    return SequentialRNN(enc, PoolingLinearClassifier(layers, drops))


emb_sz = len(embedding_weights)
emb_sz, nh, nl = 300, 300, 2
dps = tensor([0.4, 0.3, 0.4, 0.05, 0.5]) * 0.25
model = get_text_classifier(len(embedding_weights), emb_sz, nh, nl, 3, 1, bptt, *dps)

learn = Learner(databunch, model, loss_func = F.cross_entropy, metrics=accuracy)
learn.unfreeze()
learn.fit_one_cycle(1, 2e-3, moms=(0.8,0.7))

