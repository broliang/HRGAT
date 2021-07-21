import torch
from torch import nn
import dgl
import dgl.function as fn
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class RGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, act=lambda x: x, bias=True, drop_rate=0., opn='corr',
                 use_text = True, use_img = False, use_attr = False,
                 num_base=-1,
                 num_rel=None):
        super(RGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act  # activation function
        self.device = None
        self.rel = None
        self.rank = 8
        self.opn = opn
        self.use_text = use_text
        self.use_img = use_img
        self.use_attr = use_attr
        print(self.use_text,self.use_img,self.use_attr)
        self.fuse_embed = out_channels
        # if self.use_text:
        #     self.in_w_t = self.get_param([in_channels, out_channels])
        #     self.out_w_t = self.get_param([in_channels, out_channels])
        #
        #     self.text_facotr = self.get_param([self.rank, self.out_channels+1, self.out_channels])
        #     # self.fuse_embed += out_channels
        # if self.use_img:
        #     self.in_w_i = self.get_param([in_channels, out_channels])
        #     self.out_w_i = self.get_param([in_channels, out_channels])
        #     # self.loop_w_i = self.get_param([in_channels, out_channels])
        #     self.img_facotr = self.get_param([self.rank, self.out_channels+1, self.out_channels])
        #     # self.fuse_embed += out_channels
        # if self.use_attr:
        #     self.in_w_n = self.get_param([in_channels, out_channels])
        #     self.out_w_n = self.get_param([in_channels, out_channels])
        #     # self.loop_w_n = self.get_param([in_channels, out_channels])
        #     self.attr_factor = self.get_param([self.rank, self.out_channels+1, self.out_channels])
        #     # self.fuse_embed += out_channels


        # relation-type specific parameter
        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        self.w_rel = self.get_param([in_channels, out_channels])  # transform embedding of relations to next layer
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding

        self.attn_fc = nn.Linear(3 * in_channels, 1, bias= False)

        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        # self.fusion_weights = self.get_param([1, self.rank])
        # self.fusion_bias = Parameter(torch.Tensor(1, self.out_channels))
        # self.fusion_bias.data.fill_(0)
        if self.w_rel.is_cuda:
            self.DTYPE = torch.cuda.FloatTensor
        else:
            self.DTYPE = torch.FloatTensor
        if num_base > 0:
            self.rel_wt = self.get_param([num_rel * 2, num_base])
        else:
            self.rel_wt = None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges: dgl.EdgeBatch):
        edge_type = edges.data['type']  # [E, 1]
        edge_num = edge_type.shape[0]
        edge_data_h = self.comp(edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        msg = torch.cat([torch.matmul(edge_data_h[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data_h[edge_num // 2:, :], self.out_w)])
        # if self.use_text:
        #     edge_data_t = self.comp(edges.src['t'], self.rel[edge_type])  # [E, in_channel]
        #     msg_t = torch.cat([torch.matmul(edge_data_t[:edge_num // 2, :], self.in_w_t),
        #                        torch.matmul(edge_data_t[edge_num // 2:, :], self.out_w_t)])
        #     msg_t = torch.cat((Variable(torch.ones(msg_t.shape[0], 1).type(self.DTYPE).to(self.device), requires_grad=False), msg_t), dim=1)
        #     fused_msg_t = torch.matmul(msg_t, self.text_facotr)
        #
        # if self.use_img:
        #     edge_data_i = self.comp(edges.src['i'], self.rel[edge_type])  # [E, in_channel]
        #     msg_i = torch.cat([torch.matmul(edge_data_i[:edge_num // 2, :], self.in_w_i),
        #                        torch.matmul(edge_data_i[edge_num // 2:, :], self.out_w_i)])
        #     msg_i = torch.cat((Variable(torch.ones(msg_i.shape[0], 1).type(self.DTYPE).to(self.device), requires_grad=False), msg_i), dim=1)
        #     fused_msg_i = torch.matmul(msg_i, self.img_facotr)
        #
        #     # msg = torch.cat([msg, msg_i], dim=1)
        # if self.use_attr:
        #     edge_data_n = self.comp(edges.src['n'], self.rel[edge_type])  # [E, in_channel]
        #     msg_n = torch.cat([torch.matmul(edge_data_n[:edge_num // 2, :], self.in_w_n),
        #                        torch.matmul(edge_data_n[edge_num // 2:, :], self.out_w_n)])
        #     msg_n = torch.cat((Variable(torch.ones(msg_n.shape[0], 1).type(self.DTYPE).to(self.device), requires_grad=False), msg_n), dim=1)
        #     fused_msg_n = torch.matmul(msg_n, self.attr_factor)
        #
        #     # msg = torch.cat([msg, msg_n], dim=1)
        # fusion_zy = fused_msg_t * fused_msg_i * fused_msg_n
        # fused = torch.matmul(self.fusion_weights, fusion_zy.permute(1,0,2)).squeeze() + self.fusion_bias
        # msg = msg*fused
        msg = torch.nn.functional.softmax(msg)
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg, 'e': edges.data['e']}

    def reduce_func(self, nodes: dgl.NodeBatch):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = self.drop(torch.sum(alpha * nodes.mailbox['msg'], dim=1)) / 3
        return {'h': h}

    def comp(self, h, edge_data):
        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        def ccorr(a, b):
            return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')
    def edge_attention(self,edges):
        h2 = torch.cat([edges.src['h'], self.rel[edges.data['type']], edges.dst['h']], dim=1)
        a = self.attn_fc(h2)
        return {'e': F.leaky_relu(a)}

    def forward(self, g: dgl.DGLGraph, x, x_t, x_i, x_n, rel_repr, edge_type, edge_norm):
        """
        :param g: dgl Graph, a graph without self-loop
        :param x: input node features, [V, in_channel]
        :param rel_repr: input relation features: 1. not using bases: [num_rel*2, in_channel]
                                                  2. using bases: [num_base, in_channel]
        :param edge_type: edge type, [E]
        :param edge_norm: edge normalization, [E]
        :return: x: output node features: [V, out_channel]
                 rel: output relation features: [num_rel*2, out_channel]
        """
        self.device = x.device
        g = g.local_var()
        g.ndata['h'] = x
        # if self.use_text:
        #     g.ndata['t'] = x_t
        # if self.use_img:
        #     g.ndata['i'] = x_i
        # if self.use_attr:
        #     g.ndata['n'] = x_n
        g.edata['type'] = edge_type
        g.edata['norm'] = edge_norm
        if self.rel_wt is None:
            self.rel = rel_repr
        else:
            self.rel = torch.mm(self.rel_wt, rel_repr)  # [num_rel*2, num_base] @ [num_base, in_c]
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func,  self.reduce_func)
        x = g.ndata.pop('h') + torch.mm(self.comp(x, self.loop_rel), self.loop_w) / 3
        if self.bias is not None:
            x = x + self.bias
        x = self.bn(x)

        return self.act(x), torch.matmul(self.rel, self.w_rel)

# RBF Layer

class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        sigmas: the learnable scaling factors of shape (out_features).
            The values are initialised as ones.

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        # self.basis_func = self.gaussian()
        self.reset_parameters()

    def gaussian(alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.sigmas, 1)

    def forward(self, input):
        size = (input.shape[0], self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0)
        return self.gaussian(distances)


# RBFs

def gaussian(alpha):
    phi = torch.exp(-1 * alpha.pow(2))
    return phi


def linear(alpha):
    phi = alpha
    return phi


def quadratic(alpha):
    phi = alpha.pow(2)
    return phi


def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi


def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi


def poisson_two(alpha):
    phi = ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha)) \
          * alpha * torch.exp(-alpha)
    return phi


def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3 ** 0.5 * alpha) * torch.exp(-3 ** 0.5 * alpha)
    return phi


def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) \
           * alpha.pow(2)) * torch.exp(-5 ** 0.5 * alpha)
    return phi



class RelGCNCov(nn.Module):
    def __init__(self, in_channels, out_channels, act=lambda x: x, bias=True, drop_rate=0., opn='corr',
                 use_text = True, use_img = False, use_attr = False,
                 num_base=-1,
                 num_rel=None):
        super(RelGCNCov, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act  # activation function
        self.device = None
        self.rel = None
        self.rank = 8
        self.opn = opn
        self.use_text = use_text
        self.use_img = use_img
        self.use_attr = use_attr
        self.fuse_embed = out_channels

        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        self.w_rel = self.get_param([in_channels, out_channels])  # transform embedding of relations to next layer
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding



        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        # self.fusion_weights = self.get_param([1, self.rank])
        # self.fusion_bias = Parameter(torch.Tensor(1, self.out_channels))
        # self.fusion_bias.data.fill_(0)
        if self.w_rel.is_cuda:
            self.DTYPE = torch.cuda.FloatTensor
        else:
            self.DTYPE = torch.FloatTensor
        if num_base > 0:
            self.rel_wt = self.get_param([num_rel * 2, num_base])
        else:
            self.rel_wt = None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges: dgl.EdgeBatch):
        edge_type = edges.data['type']  # [E, 1]
        edge_num = edge_type.shape[0]
        edge_data_h = self.rel[edge_type]
        msg = torch.cat([torch.matmul(edge_data_h[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data_h[edge_num // 2:, :], self.out_w)])
        msg = torch.nn.functional.softmax(msg)
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    def reduce_func(self, nodes: dgl.NodeBatch):
        return {'h': self.drop(nodes.data['h']) / 3}

    def comp(self, h, edge_data):
        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        def ccorr(a, b):
            return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: dgl.DGLGraph, x, rel_repr, edge_type, edge_norm):
        """
        :param g: dgl Graph, a graph without self-loop
        :param x: input node features, [V, in_channel]
        :param rel_repr: input relation features: 1. not using bases: [num_rel*2, in_channel]
                                                  2. using bases: [num_base, in_channel]
        :param edge_type: edge type, [E]
        :param edge_norm: edge normalization, [E]
        :return: x: output node features: [V, out_channel]
                 rel: output relation features: [num_rel*2, out_channel]
        """
        self.device = x.device
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = edge_type
        g.edata['norm'] = edge_norm
        if self.rel_wt is None:
            self.rel = rel_repr
        else:
            self.rel = torch.mm(self.rel_wt, rel_repr)  # [num_rel*2, num_base] @ [num_base, in_c]
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        x = g.ndata.pop('h') + torch.mm(self.comp(x, self.loop_rel), self.loop_w) / 3
        if self.bias is not None:
            x = x + self.bias
        x = self.bn(x)

        return self.act(x), torch.matmul(self.rel, self.w_rel)



if __name__ == '__main__':
    RGAT = RGATConv(in_channels=10, out_channels=5)
    src, tgt = [0, 1, 0, 3, 2], [1, 3, 3, 4, 4]
    g = dgl.DGLGraph()
    g.add_nodes(5)
    g.add_edges(src, tgt)  # src -> tgt
    g.add_edges(tgt, src)  # tgt -> src
    edge_type = torch.tensor([0, 0, 0, 1, 1] + [2, 2, 2, 3, 3])
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = in_deg ** -0.5
    norm[np.isinf(norm)] = 0
    g.ndata['xxx'] = norm
    g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
    edge_norm = g.edata.pop('xxx').squeeze()

    x = torch.randn([5, 10])
    rel = torch.randn([4, 10])  # 2*2+1
    x, rel = RGAT(g, x, rel, edge_type, edge_norm)
    print(x.shape, rel.shape)
