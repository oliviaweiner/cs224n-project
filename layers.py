"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob, char_vectors):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors)
        self.proj = nn.Linear(word_vectors.size(1) + char_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)
        # print('started init')
        # self.conv_layer = nn.Conv2d(in_channels=char_vectors.size(1), out_channels=char_vectors.size(1), kernel_size=(0, 5))
        # print('completed layer')
        #- set so that hin and Hout are the same
        #in channels is number of input dimensions in each character embedding
        #out chanels should be same if not changing size (hidden state hidden_size)
        #stride 1
        #width is max length of any word in n. of characters. In args documentation
        #width specified on p. 6 of paper, they use width of 5, width out is 1
        self.hidden_size = hidden_size

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        seq_len = emb.size(1)
        print('started init')
        self.conv_layer = nn.Conv2d(in_channels=seq_len, out_channels=seq_len, kernel_size=(1, 4))
        print('completed layer')
        char_emb = self.char_embed(x) # (batch_size, seq_len, char_embed_size)
        print('char_embed size:')
        print(char_emb.size(2))
        char_emb = char_emb.view(char_emb.size(0), int(char_emb.size(2)/4), char_emb.size(1), 4)
        print('started forward')
        char_emb = self.conv_layer(char_emb)
        print('used conv layer')
        cat_emb = torch.cat((emb, char_emb), dim=2) # (batch_size, seq_len, embed_size + char_embed_size)
        cat_emb = F.dropout(cat_emb, self.drop_prob, self.training)
        cat_emb = self.proj(cat_emb)  # (batch_size, seq_len, hidden_size)
        cat_emb = self.hwy(cat_emb)   # (batch_size, seq_len, hidden_size)

        return cat_emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2



class CoAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, num_layers=1, drop_prob=0.1):
        super(CoAttention, self).__init__()
        self.drop_prob = drop_prob
        self.q_prime_weight = nn.Parameter(torch.zeros(1, hidden_size, hidden_size)) #our edit
        self.q_prime_bias = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.c_null = nn.Parameter(torch.zeros(1,1,hidden_size))
        self.q_prime_null = nn.Parameter(torch.zeros(1,1,hidden_size))
        self.u_biLSTM = nn.LSTM(2 * hidden_size, 2 * hidden_size, num_layers=num_layers, batch_first=True, dropout=drop_prob, bidirectional=True)
        for weight in (self.c_null, self.q_prime_null, self.q_prime_weight, self.q_prime_bias): #added q_prime_weight
            nn.init.xavier_uniform_(weight)

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        q_prime = torch.bmm(q, self.q_prime_weight.expand(batch_size, -1, -1))
        q_prime += self.q_prime_bias
        q_prime = torch.tanh(q_prime)   # (batch_size, M, hidden_size)
        q_prime = torch.cat((q_prime, self.q_prime_null.expand(batch_size, -1, -1)), dim=1) # (batch_size, M+1, hidden_size)
        c = torch.cat((c, self.c_null.expand(batch_size, -1, -1)), dim=1) # (batch_size, N+1, hidden_size)
        L = torch.bmm(c, torch.transpose(q_prime, 1, 2))  # (batch_size, N+1, M+1)
        ones = torch.ones(batch_size, 1, 1)
        ones = ones.to(c_mask.device)
        c_mask = torch.cat((c_mask.view(batch_size, c_len, 1), ones), dim=1)  # (batch_size, c_len, 1)
        q_mask = torch.cat((q_mask.view(batch_size, 1, q_len), ones), dim=2) # batch_size, 1, q_len
        alpha = masked_softmax(L, q_mask, dim=2)  # (batch_size, N+1, M+1)
        a = torch.bmm(alpha, q_prime)  # (batch_size, N+1, hidden_size)
        beta = masked_softmax(L, c_mask, dim=1)  # (batch_size, N+1, M+1)
        b = torch.bmm(torch.transpose(beta, 1, 2), c) # (batch_size, M+1, hidden_size)
        s = torch.bmm(alpha, b) # (batch_size, N+1, hidden_size)
        u = torch.split(torch.cat((s, a), dim=2), [c_len, 1], dim=1)[0]  # (batch_size, N+1, 2*hidden_size)
        self.u_biLSTM.flatten_parameters()
        x, get_rid_of = self.u_biLSTM(u) # (batch_size, N+1, 4*hidden_size)

        return x

class HighwayMaxoutNetwork(nn.Module):
    def __init__(self, hidden_size, maxout_options, drop_prob = 0.1):
        super(HighwayMaxoutNetwork, self).__init__()
        self.w_d = nn.Parameter(torch.zeros(1, 5 * hidden_size, hidden_size))
        self.w_1 = nn.Parameter(torch.zeros(maxout_options, 3*hidden_size, hidden_size))
        self.b_1 = nn.Parameter(torch.zeros(1, maxout_options, 1, hidden_size))
        self.w_2 = nn.Parameter(torch.zeros(maxout_options, hidden_size, hidden_size))
        self.b_2 = nn.Parameter(torch.zeros(1, maxout_options, 1, hidden_size))
        self.w_3 = nn.Parameter(torch.zeros(maxout_options, 2*hidden_size, 1))
        self.b_3 = nn.Parameter(torch.zeros(1, maxout_options, 1, 1))
        self.dropout = nn.Dropout(p=drop_prob)
        for weight in (self.w_d, self.w_1, self.b_1, self.w_2, self.b_2, self.w_3, self.b_3):
            nn.init.xavier_uniform_(weight)

    def forward(self, h_vec, coattention):
        r = torch.tanh(h_vec @ self.w_d).expand(-1, coattention.size(1), -1)  # (b,m,l)
        m_1 = torch.cat((coattention, r), dim=2).unsqueeze(1) #(b,1,m,3l)
        m_1 = (m_1 @ self.w_1) + self.b_1 #(b,p,m,l)
        m_1 = self.dropout(m_1)  # (b,p,m,l)
        m_1 = torch.max(m_1, 1, keepdim=True)[0] #(b,1,m,l)
        m_2 = (m_1 @ self.w_2) + self.b_2 #(b,p,m,l)
        m_2 = self.dropout(m_2)  # (b,p,m,l)
        m_2 = torch.max(m_2, 1, keepdim=True)[0] #(b,1,m,l)
        out = torch.cat((m_1, m_2), dim=3) #(b,1,m,2l)
        out = (out @ self.w_3) + self.b_3 #(b,p,m,1)
        out = self.dropout(out.squeeze(dim=3)) #(b,p,m)
        out = torch.max(out, dim=1)[0] #(b,m)

        return out


class Decoder(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, maxout_options, drop_prob=0.1):
        super(Decoder, self).__init__()
        self.coatt_length = 2 * hidden_size
        self.LSTM_dec = nn.LSTM(input_size=4*hidden_size, hidden_size=hidden_size, batch_first=True, dropout=drop_prob)
        self.HMN_start = HighwayMaxoutNetwork(hidden_size, maxout_options)
        self.HMN_end = HighwayMaxoutNetwork(hidden_size, maxout_options)

    def forward(self, h, c, start_predictions, end_predictions, coattention, c_mask):
        #h => (b, 1, l)
        #coattention => (b, m, 2l)

        start_encoding = torch.gather(coattention, 1, start_predictions.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,self.coatt_length)) #(b, 1, 2l)
        end_encoding = torch.gather(coattention, 1, end_predictions.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,self.coatt_length)) #(b, 1, 2l)
        self.LSTM_dec.flatten_parameters()
        if type(h) != type(None):
            get_rid_of, (new_h, new_c) = self.LSTM_dec(torch.cat((start_encoding, end_encoding), dim=2), (h, c)) #(b, 1, l)
        else:
            get_rid_of, (new_h, new_c) = self.LSTM_dec(torch.cat((start_encoding, end_encoding), dim=2))
        h_vec = torch.cat((new_h.transpose(0, 1), start_encoding, end_encoding), dim=2) #(b, 1, 5l)
        alphas = masked_softmax(self.HMN_start(h_vec, coattention), c_mask, log_softmax=True)
        betas = masked_softmax(self.HMN_end(h_vec, coattention), c_mask, log_softmax=True)

        return new_h, new_c, alphas, betas

class DynamicDecoder(nn.Module):
    def __init__(self, hidden_size, maxout_options, drop_prob=0.1):
        super(DynamicDecoder, self).__init__()
        self.decoder = Decoder(hidden_size, maxout_options, drop_prob)

    def forward(self, c_len, c_mask, coattention):
        h = None
        c = None
        start_predictions = torch.zeros_like(c_len)
        end_predictions = c_len - torch.ones_like(c_len)
        cumulative_alphas = torch.zeros_like(c_mask, dtype=torch.float)
        cumulative_betas = torch.zeros_like(c_mask, dtype=torch.float)
        iters = 2
        for i in range(iters):
            h, c, alphas, betas = self.decoder(h, c, start_predictions, end_predictions, coattention, c_mask)
            cumulative_alphas += alphas
            cumulative_betas += betas
            start_predictions = torch.max(alphas, 1)[1]
            end_predictions = torch.max(betas, 1)[1]
        out = cumulative_alphas / iters, cumulative_betas / iters, start_predictions, end_predictions

        return out