# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFVIDependency(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals
    of dependency trees (:cite:`wang-tu-2020-second`).
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of three tensors `s_arc`, `s_sib` and `s_grd`.
                `s_arc` (``[batch_size, seq_len, seq_len]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        logQ = self.mfvi(*scores, mask)
        marginals = logQ.exp()

        if target is None:
            return marginals
        loss = F.nll_loss(logQ[mask], target[mask])

        return loss, marginals

    def mfvi(self, s_arc, s_sib, mask):
        batch_size, seq_len = mask.shape
        ls, rs = torch.stack(torch.where(mask.new_ones(seq_len, seq_len))).view(-1, seq_len, seq_len).contiguous().sort(0)[0]
        mask = mask.index_fill(1, ls.new_tensor(0), 1)
        # [seq_len, seq_len, batch_size], (h->m)
        mask = (mask.unsqueeze(-1) & mask.unsqueeze(-2)).permute(2, 1, 0).contiguous()
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & ls.unsqueeze(-1).ne(ls.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & rs.unsqueeze(-1).ne(rs.new_tensor(range(seq_len))).unsqueeze(-1)
        # [seq_len, seq_len, batch_size], (h->m)
        s_arc = s_arc.permute(2, 1, 0).contiguous()
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0).contiguous() * mask2o

        # posterior distributions
        # [seq_len, seq_len, batch_size], (h->m)
        q = s_arc

        for _ in range(self.max_iter):
            q = q.softmax(0)
            # q(ij) = s(ij) + sum(q(ik)s^sib(ij,ik)), k != i,j
            q = s_arc + (q.unsqueeze(1) * s_sib).sum(2)

        return q.permute(2, 1, 0).contiguous().log_softmax(-1)


class MFVIConstituency(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals
    of dependency trees (:cite:`wang-tu-2020-second`).
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of two tensors `s_arc` and `s_sib`.
                `s_arc` (``[batch_size, seq_len, seq_len]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        marginals = self.mfvi(*scores, mask)

        if target is None:
            return marginals
        loss = F.binary_cross_entropy(marginals[mask], target[mask].float())

        return loss, marginals

    def mfvi(self, s_span, s_pair, mask):
        batch_size, seq_len, _ = mask.shape
        ls, rs = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len).contiguous().sort(0)[0]
        # [seq_len, seq_len, batch_size], (l->r)
        mask = mask.permute(1, 2, 0).contiguous()
        # [seq_len, seq_len, seq_len, batch_size], (l->r->b)
        mask2o = mask.unsqueeze(2).repeat(1, 1, seq_len, 1)
        mask2o = mask2o & ls.unsqueeze(-1).ne(ls.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & rs.unsqueeze(-1).ne(rs.new_tensor(range(seq_len))).unsqueeze(-1)
        # [seq_len, seq_len, batch_size], (l->r)
        s_span = s_span.permute(1, 2, 0).contiguous()
        # [seq_len, seq_len, seq_len, batch_size], (l->r->b)
        s_pair = s_pair.permute(1, 2, 3, 0).contiguous() * mask2o

        # posterior distributions
        # [seq_len, seq_len, batch_size], (l->r)
        q = s_span

        for _ in range(self.max_iter):
            q = q.sigmoid()
            # q(ij) = s(ij) + sum(q(jk)*s^bin(ij,jk), k != i,j
            q = s_span + (q.unsqueeze(1) * s_pair).sum(2)

        return q.permute(2, 0, 1).contiguous().sigmoid()


class MFVISemanticDependency(nn.Module):
    r"""
    Mean Field Variational Inference for approximately calculating marginals
    of semantic dependency trees (:cite:`wang-etal-2019-second`).
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of four tensors `s_edge`, `s_sib`, `s_cop` and `s_grd`.
                `s_edge` (``[batch_size, seq_len, seq_len]``) holds scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
                `s_cop` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-coparent triples.
                `s_grd` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-grandparent triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.
        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        logits = self.mfvi(*scores, mask)
        marginals = logits.sigmoid()

        if target is None:
            return marginals
        loss = F.binary_cross_entropy_with_logits(logits[mask], target[mask].float())

        return loss, marginals

    def mfvi(self, s_edge, s_sib, s_cop, s_grd, mask):
        batch_size, seq_len, _ = mask.shape
        hs, ms = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len).contiguous()
        # [seq_len, seq_len, batch_size], (h->m)
        mask = mask.permute(2, 1, 0).contiguous()
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len))).unsqueeze(-1)
        # [seq_len, seq_len, batch_size], (h->m)
        s_edge = s_edge.permute(2, 1, 0).contiguous()
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0).contiguous() * mask2o
        # [seq_len, seq_len, seq_len, batch_size], (h->m->c)
        s_cop = s_cop.permute(2, 1, 3, 0).contiguous() * mask2o
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        s_grd = s_grd.permute(2, 1, 3, 0).contiguous() * mask2o

        # posterior distributions
        # [seq_len, seq_len, batch_size], (h->m)
        q = s_edge

        for _ in range(self.max_iter):
            q = q.sigmoid()
            # q(ij) = s(ij) + sum(q(ik)s^sib(ij,ik) + q(kj)s^cop(ij,kj) + q(jk)s^grd(ij,jk)), k != i,j
            q = s_edge + (q.unsqueeze(1) * s_sib + q.transpose(0, 1).contiguous().unsqueeze(0) * s_cop + q.unsqueeze(0) * s_grd).sum(2)

        return q.permute(2, 1, 0).contiguous()


class LBPSemanticDependency(nn.Module):
    r"""
    Loopy Belief Propagation for approximately calculating marginals
    of semantic dependency trees (:cite:`wang-etal-2019-second`).
    """

    def __init__(self, max_iter=3):
        super().__init__()

        self.max_iter = max_iter

    def __repr__(self):
        return f"{self.__class__.__name__}(max_iter={self.max_iter})"

    @torch.enable_grad()
    def forward(self, scores, mask, target=None):
        r"""
        Args:
            scores (~torch.Tensor, ~torch.Tensor):
                Tuple of four tensors `s_edge`, `s_sib`, `s_cop` and `s_grd`.
                `s_edge` (``[batch_size, seq_len, seq_len]``) holds Scores of all possible dependent-head pairs.
                `s_sib` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-sibling triples.
                `s_cop` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-coparent triples.
                `s_grd` (``[batch_size, seq_len, seq_len, seq_len]``) holds the scores of dependent-head-grandparent triples.
            mask (~torch.BoolTensor): ``[batch_size, seq_len, seq_len]``.
                The mask to avoid aggregation on padding tokens.
            target (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                A Tensor of gold-standard dependent-head pairs. Default: ``None``.

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor for marginals of shape ``[batch_size, seq_len, seq_len]``.
        """

        logits = self.lbp(*scores, mask)
        marginals = logits.softmax(-1)[..., 1]

        if target is None:
            return marginals
        loss = F.cross_entropy(logits[mask], target[mask])

        return loss, marginals

    def lbp(self, s_edge, s_sib, s_cop, s_grd, mask):
        batch_size, seq_len, _ = mask.shape
        hs, ms = torch.stack(torch.where(torch.ones_like(mask[0]))).view(-1, seq_len, seq_len).contiguous()
        # [seq_len, seq_len, batch_size], (h->m)
        mask = mask.permute(2, 1, 0).contiguous()
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        mask2o = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask2o = mask2o & hs.unsqueeze(-1).ne(hs.new_tensor(range(seq_len))).unsqueeze(-1)
        mask2o = mask2o & ms.unsqueeze(-1).ne(ms.new_tensor(range(seq_len))).unsqueeze(-1)
        # [2, seq_len, seq_len, batch_size], (h->m)
        s_edge = torch.stack((torch.zeros_like(s_edge), s_edge)).permute(0, 3, 2, 1).contiguous()
        # [seq_len, seq_len, seq_len, batch_size], (h->m->s)
        s_sib = s_sib.permute(2, 1, 3, 0).contiguous()
        # [seq_len, seq_len, seq_len, batch_size], (h->m->c)
        s_cop = s_cop.permute(2, 1, 3, 0).contiguous()
        # [seq_len, seq_len, seq_len, batch_size], (h->m->g)
        s_grd = s_grd.permute(2, 1, 3, 0).contiguous()

        # log beliefs
        # [2, seq_len, seq_len, batch_size], (h->m)
        q = s_edge
        # log messages of siblings
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->s)
        m_sib = s_sib.new_zeros(2, seq_len, seq_len, seq_len, batch_size)
        # log messages of co-parents
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->c)
        m_cop = s_cop.new_zeros(2, seq_len, seq_len, seq_len, batch_size)
        # log messages of grand-parents
        # [2, seq_len, seq_len, seq_len, batch_size], (h->m->g)
        m_grd = s_grd.new_zeros(2, seq_len, seq_len, seq_len, batch_size)

        for _ in range(self.max_iter):
            q = q.log_softmax(0)
            # m(ik->ij) = logsumexp(q(ik) - m(ij->ik) + s(ij->ik))
            m = q.unsqueeze(3) - m_sib
            m_sib = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + s_sib)).logsumexp(0))).log_softmax(0)
            m = q.unsqueeze(3) - m_cop
            m_cop = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + s_cop)).logsumexp(0))).log_softmax(0)
            m = q.unsqueeze(3) - m_grd
            m_grd = torch.stack((m.logsumexp(0), torch.stack((m[0], m[1] + s_grd)).logsumexp(0))).log_softmax(0)
            # q(ij) = s(ij) + sum(m(ik->ij)), k != i,j
            q = s_edge + ((m_sib + m_cop + m_grd).transpose(2, 3).contiguous() * mask2o).sum(3)

        return q.permute(3, 2, 1, 0).contiguous()