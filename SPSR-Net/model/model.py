from torch import nn
from transformers import AutoModel
from fastNLP import seq_len_to_mask
try:
    from torch_scatter import scatter_max
except ImportError:
    # The project only needs max pooling over subword positions. PyTorch 2.x
    # provides the equivalent operation, so keep training runnable when a
    # torch-scatter wheel is unavailable for the host CUDA/PyTorch build.
    def scatter_max(src, index, dim=1):
        if dim != 1:
            raise NotImplementedError('The torch_scatter fallback only supports dim=1')
        expanded_index = index.unsqueeze(-1).expand_as(src)
        output_size = int(index.max().item()) + 1
        # ``torch_scatter.scatter_max`` leaves unused index positions at zero.
        # Padding rows have no subword assigned to them; using -inf here lets
        # them reach the unmasked score branch and creates NaN in BCE (0 * NaN
        # is still NaN) even though those rows are later ignored by the label
        # mask. Zero keeps the fallback numerically equivalent and padding-safe.
        output = src.new_zeros((src.size(0), output_size, src.size(2)))
        output.scatter_reduce_(1, expanded_index, src, reduce='amax', include_self=True)
        return output, None
import torch
import torch.nn.functional as F
from .cnn import MaskCNN_1,MaskCNN_2
from .multi_head_biaffine3 import MultiHeadBiaffine


class ResidualMLPScoreHead(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, hidden_ratio=1.5):
        super(ResidualMLPScoreHead, self).__init__()
        hidden_dim = max(int(in_features * hidden_ratio), out_features)
        self.base = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(in_features)
        self.up = nn.Linear(in_features, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.down = nn.Linear(hidden_dim, out_features)
        self.res_scale = nn.Parameter(torch.tensor(0.0))

        torch.nn.init.xavier_normal_(self.base.weight.data)
        torch.nn.init.xavier_normal_(self.up.weight.data)
        torch.nn.init.xavier_normal_(self.down.weight.data)
        if self.base.bias is not None:
            torch.nn.init.zeros_(self.base.bias.data)
        if self.up.bias is not None:
            torch.nn.init.zeros_(self.up.bias.data)
        if self.down.bias is not None:
            torch.nn.init.zeros_(self.down.bias.data)

    def forward(self, x):
        base = self.base(x)
        y = self.norm(x)
        y = self.up(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.down(y)
        return base + torch.tanh(self.res_scale) * y


class RelationAwareSpanScoreHead(nn.Module):
    """Classify spans after exchanging information with boundary-sharing spans.

    The input is the HSR-enhanced span map ``[batch, start, end, feature]``.
    For a candidate (i, j), the head attends separately to candidates (i, k)
    with the same left boundary and (k, j) with the same right boundary.  Such
    candidates form natural extension/containment chains in nested NER, while
    keeping the cost linear in the number of start/end groups rather than
    attending over every pair of spans in a sentence.

    ``relation_scale`` starts at zero.  Consequently, loading an existing
    ordinary classifier checkpoint produces exactly its original logits at the
    first step, and the relation correction is learned gradually.
    """

    def __init__(self, in_features, out_features, relation_dim=None, dropout=0.1):
        super().__init__()
        relation_dim = relation_dim or max(64, in_features // 4)
        self.base = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(in_features)
        self.q_proj = nn.Linear(in_features, relation_dim, bias=False)
        self.k_proj = nn.Linear(in_features, relation_dim, bias=False)
        self.v_proj = nn.Linear(in_features, relation_dim, bias=False)
        self.fuse = nn.Sequential(
            nn.Linear(in_features + relation_dim * 2, in_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features, out_features),
        )
        self.gate = nn.Linear(in_features + relation_dim * 2, out_features)
        self.relation_scale = nn.Parameter(torch.tensor(0.0))

        for module in (self.base, self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_normal_(module.weight)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)
        for module in self.fuse:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_normal_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @staticmethod
    def _masked_attention(query, key, value, key_mask):
        """Attention over one boundary-sharing span chain.

        Inputs are ``[groups, spans, dim]`` and key_mask is ``[groups, spans]``.
        Invalid lower-triangular/padding entries are never used as keys.
        """
        scale = query.size(-1) ** -0.5
        logits = torch.bmm(query, key.transpose(1, 2)) * scale
        logits = logits.masked_fill(~key_mask.unsqueeze(1), -1e4)
        attention = torch.softmax(logits, dim=-1)
        return torch.bmm(attention, value)

    def forward(self, x, word_lengths):
        batch_size, length, _, _ = x.shape
        base_logits = self.base(x)

        # A span is valid only in the upper triangle and inside its sentence.
        positions = torch.arange(length, device=x.device)
        token_mask = positions.unsqueeze(0) < word_lengths.unsqueeze(1)
        upper = positions.view(length, 1) <= positions.view(1, length)
        valid_span = token_mask.unsqueeze(2) & token_mask.unsqueeze(1) & upper.unsqueeze(0)

        normed = self.norm(x)
        query = self.q_proj(normed)
        key = self.k_proj(normed)
        value = self.v_proj(normed)

        # Same-start chains: (i, j) attends to (i, k).
        row_context = self._masked_attention(
            query.reshape(batch_size * length, length, -1),
            key.reshape(batch_size * length, length, -1),
            value.reshape(batch_size * length, length, -1),
            valid_span.reshape(batch_size * length, length),
        ).reshape(batch_size, length, length, -1)

        # Same-end chains: (i, j) attends to (k, j).
        col_context = self._masked_attention(
            query.transpose(1, 2).reshape(batch_size * length, length, -1),
            key.transpose(1, 2).reshape(batch_size * length, length, -1),
            value.transpose(1, 2).reshape(batch_size * length, length, -1),
            valid_span.transpose(1, 2).reshape(batch_size * length, length),
        ).reshape(batch_size, length, length, -1).transpose(1, 2)

        fused = torch.cat([normed, row_context, col_context], dim=-1)
        correction = torch.sigmoid(self.gate(fused)) * self.fuse(fused)
        return base_logits + torch.tanh(self.relation_scale) * correction


class LengthGatedSpanScoreHead(nn.Module):
    """Dynamically balance raw, SNSA and HSR features by span length.

    The input feature order is fixed by ``CNNNer.forward``: raw span features,
    SNSA-enhanced features, then HSR features.  A length-conditioned gate gives
    each candidate its own three-way mixture.  It starts as an exact ordinary
    linear classifier (uniform gates are rescaled by three), which makes the
    fusion learnable without destabilizing the established SPSR-Net backbone.
    """

    def __init__(self, branch_dim, out_features, length_bins=8, dropout=0.1):
        super().__init__()
        self.branch_dim = branch_dim
        self.length_bins = length_bins
        length_dim = max(16, branch_dim // 8)
        self.length_embedding = nn.Embedding(length_bins, length_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(branch_dim * 3 + length_dim),
            nn.Linear(branch_dim * 3 + length_dim, branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, 3),
        )
        self.classifier = nn.Linear(branch_dim * 3, out_features)
        nn.init.xavier_normal_(self.length_embedding.weight)
        nn.init.xavier_normal_(self.gate[1].weight)
        nn.init.zeros_(self.gate[1].bias)
        # Exact uniform mixture at initialization: no initial disturbance to
        # the ordinary span-classification path.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        batch_size, length, _, features = x.shape
        if features != self.branch_dim * 3:
            raise ValueError(f'Expected {self.branch_dim * 3} span features, got {features}')
        row_ids = torch.arange(length, device=x.device).view(length, 1)
        col_ids = torch.arange(length, device=x.device).view(1, length)
        span_lengths = (col_ids - row_ids).clamp(min=0, max=self.length_bins - 1)
        length_features = self.length_embedding(span_lengths).unsqueeze(0).expand(batch_size, -1, -1, -1)
        gate_logits = self.gate(torch.cat([x, length_features], dim=-1))
        # Multiplying softmax weights by 3 retains x exactly under the initial
        # uniform gate and lets each branch be strengthened or suppressed later.
        gates = torch.softmax(gate_logits, dim=-1) * 3.0
        gated = torch.cat(
            [
                x[..., :self.branch_dim] * gates[..., 0:1],
                x[..., self.branch_dim:self.branch_dim * 2] * gates[..., 1:2],
                x[..., self.branch_dim * 2:] * gates[..., 2:3],
            ],
            dim=-1,
        )
        return self.classifier(gated)


class CNNNer(nn.Module):
    def __init__(self, model_name, num_ner_tag, cnn_dim=200, biaffine_size=200,
                 size_embed_dim=0, logit_drop=0, kernel_size=3, n_head=4, cnn_depth=3, n_layer=2,
                 separateness_rate=0.1, theta=1, loss_theta=1, sad_topk=2,
                 sad_attn_dim=None, use_sad=True, use_hsr=True,
                 sad_use_rel_bias=True, sad_gate=True,
                 head_type='linear', loss_type='bce', asl_gamma_pos=0.0,
                 asl_gamma_neg=3.0, asl_clip=0.05, balanced_neg_weight=1.0,
                 label_loss_weights=None, use_length_bias=False,
                 length_bias_bins=6, bhpc_weight=0.0, bhpc_dim=128,
                 bhpc_temperature=0.1, bhpc_momentum=0.95, bhpc_margin=0.1,
                 bhpc_label_weights=None, bhpc_warmup_steps=0,
                 bhpc_prototype_scale=1.0, bhpc_boundary_scale=1.0,
                 relation_dim=None, boundary_aux_weight=0.0,
                 boundary_align_weight=0.0, conflict_rank_weight=0.0,
                 conflict_margin=0.2, conflict_warmup_steps=0,
                 length_gate_bins=8, quality_aux_weight=0.0,
                 quality_min_iou=0.1, hierarchy_aux_weight=0.0,
                 difficulty_curriculum=False, curriculum_warmup_steps=0,
                 curriculum_ramp_steps=1, curriculum_nested_boost=0.5,
                 curriculum_long_boost=0.25, curriculum_hard_neg_boost=0.25,
                 curriculum_long_span=4):
        super(CNNNer, self).__init__()
        self.mdim =(cnn_dim) 
        self.num_ner_tag = num_ner_tag
        self.cnn_dim = cnn_dim
        self.separateness_rate = separateness_rate
        self.cnn_dim = cnn_dim
        self.loss_theta = loss_theta
        self.n_layer = n_layer
        if head_type not in ('linear', 'residual_mlp', 'relation_aware', 'length_gated'):
            raise ValueError(
                "head_type must be one of ['linear', 'residual_mlp', 'relation_aware', 'length_gated'], "
                f"got {head_type}"
            )
        if loss_type not in ('bce', 'asl', 'balanced_asl'):
            raise ValueError(f"loss_type must be one of ['bce', 'asl', 'balanced_asl'], got {loss_type}")
        self.head_type = head_type
        self.loss_type = loss_type
        self.asl_gamma_pos = asl_gamma_pos
        self.asl_gamma_neg = asl_gamma_neg
        self.asl_clip = asl_clip
        self.balanced_neg_weight = balanced_neg_weight
        self.use_length_bias = use_length_bias
        self.length_bias_bins = length_bias_bins
        self.bhpc_weight = float(bhpc_weight)
        self.bhpc_temperature = float(bhpc_temperature)
        self.bhpc_momentum = float(bhpc_momentum)
        self.bhpc_margin = float(bhpc_margin)
        self.bhpc_warmup_steps = int(bhpc_warmup_steps)
        self.bhpc_prototype_scale = float(bhpc_prototype_scale)
        self.bhpc_boundary_scale = float(bhpc_boundary_scale)
        self.boundary_aux_weight = float(boundary_aux_weight)
        self.boundary_align_weight = float(boundary_align_weight)
        self.conflict_rank_weight = float(conflict_rank_weight)
        self.conflict_margin = float(conflict_margin)
        self.conflict_warmup_steps = int(conflict_warmup_steps)
        self.quality_aux_weight = float(quality_aux_weight)
        self.quality_min_iou = float(quality_min_iou)
        self.hierarchy_aux_weight = float(hierarchy_aux_weight)
        self.difficulty_curriculum = bool(difficulty_curriculum)
        self.curriculum_warmup_steps = int(curriculum_warmup_steps)
        self.curriculum_ramp_steps = int(curriculum_ramp_steps)
        self.curriculum_nested_boost = float(curriculum_nested_boost)
        self.curriculum_long_boost = float(curriculum_long_boost)
        self.curriculum_hard_neg_boost = float(curriculum_hard_neg_boost)
        self.curriculum_long_span = int(curriculum_long_span)
        if self.bhpc_weight < 0:
            raise ValueError(f'bhpc_weight must be >= 0, got {bhpc_weight}')
        if self.bhpc_temperature <= 0:
            raise ValueError(f'bhpc_temperature must be > 0, got {bhpc_temperature}')
        if not 0 <= self.bhpc_momentum < 1:
            raise ValueError(f'bhpc_momentum must be in [0, 1), got {bhpc_momentum}')
        if self.bhpc_margin < 0:
            raise ValueError(f'bhpc_margin must be >= 0, got {bhpc_margin}')
        if self.bhpc_warmup_steps < 0:
            raise ValueError(f'bhpc_warmup_steps must be >= 0, got {bhpc_warmup_steps}')
        if self.bhpc_prototype_scale < 0 or self.bhpc_boundary_scale < 0:
            raise ValueError('BHPC loss scales must be >= 0')
        if self.boundary_aux_weight < 0 or self.boundary_align_weight < 0:
            raise ValueError('Boundary auxiliary-loss scales must be >= 0')
        if self.conflict_rank_weight < 0 or self.conflict_margin < 0:
            raise ValueError('Conflict-ranking loss weight and margin must be >= 0')
        if self.conflict_warmup_steps < 0:
            raise ValueError('conflict_warmup_steps must be >= 0')
        if self.quality_aux_weight < 0:
            raise ValueError('quality_aux_weight must be >= 0')
        if not 0 < self.quality_min_iou <= 1:
            raise ValueError('quality_min_iou must be in (0, 1]')
        if self.hierarchy_aux_weight < 0:
            raise ValueError('hierarchy_aux_weight must be >= 0')
        if self.curriculum_warmup_steps < 0 or self.curriculum_ramp_steps < 1:
            raise ValueError('Curriculum warmup must be >= 0 and ramp steps must be >= 1')
        if min(self.curriculum_nested_boost, self.curriculum_long_boost, self.curriculum_hard_neg_boost) < 0:
            raise ValueError('Curriculum difficulty boosts must be >= 0')
        if self.curriculum_long_span < 1:
            raise ValueError('curriculum_long_span must be >= 1')
        if label_loss_weights is None:
            label_loss_weights = torch.ones(num_ner_tag, dtype=torch.float32)
        else:
            label_loss_weights = torch.as_tensor(label_loss_weights, dtype=torch.float32)
            if label_loss_weights.numel() != num_ner_tag:
                raise ValueError(
                    f"label_loss_weights must have {num_ner_tag} values, got {label_loss_weights.numel()}"
                )
        self.register_buffer('label_loss_weights', label_loss_weights.view(1, 1, 1, num_ner_tag))
        # self.param_span= nn.Parameter(torch.randn(2,cnn_dim)/20,requires_grad=True)
        self.pretrain_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.pretrain_model.config.hidden_size
        if size_embed_dim != 0:
            n_pos = 30
            self.size_embedding = torch.nn.Embedding(n_pos, size_embed_dim)
            _span_size_ids = torch.arange(512) - torch.arange(512).unsqueeze(-1)
            _span_size_ids.masked_fill_(_span_size_ids < -n_pos / 2, -n_pos / 2)
            _span_size_ids = _span_size_ids.masked_fill(_span_size_ids >= n_pos / 2, n_pos / 2 - 1) + n_pos / 2
            self.register_buffer('span_size_ids', _span_size_ids.long())
            hsz = biaffine_size * 2 + size_embed_dim + 2 
        else:
            hsz = biaffine_size * 2 + 2
        biaffine_input_size = hidden_size
        self.dropout = nn.Dropout(logit_drop)
        self.dropout1 = nn.Dropout(logit_drop)
        self.head_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.GELU(),
        )
        self.tail_mlp = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(biaffine_input_size, biaffine_size),
            nn.GELU(),
        )
        if n_head > 0:
            self.biaffine = MultiHeadBiaffine(biaffine_size, cnn_dim, n_head=n_head)
        else:
            self.U = nn.Parameter(torch.randn(cnn_dim, biaffine_size, biaffine_size))
            torch.nn.init.xavier_normal_(self.U.data)
        self.W = torch.nn.Parameter(torch.empty(cnn_dim, hsz))
        torch.nn.init.xavier_normal_(self.W.data)
        if cnn_depth > 0:
            if self.n_layer == 1:
                self.cnn1 = MaskCNN_1(
                    cnn_dim,
                    cnn_dim,
                    kernel_size=kernel_size,
                    depth=cnn_depth,
                    theta=theta,
                    sad_topk=sad_topk,
                    sad_attn_dim=sad_attn_dim,
                    use_sad=use_sad,
                    use_hsr=use_hsr,
                    sad_use_rel_bias=sad_use_rel_bias,
                    sad_gate=sad_gate,
                )
            elif self.n_layer == 2:
                self.cnn1 = MaskCNN_2(
                    cnn_dim,
                    cnn_dim,
                    kernel_size=kernel_size,
                    depth=cnn_depth,
                    theta=theta,
                    sad_topk=sad_topk,
                    sad_attn_dim=sad_attn_dim,
                    use_sad=use_sad,
                    use_hsr=use_hsr,
                    sad_use_rel_bias=sad_use_rel_bias,
                    sad_gate=sad_gate,
                )
        if self.head_type == 'residual_mlp':
            self.score_head = ResidualMLPScoreHead(
                in_features=cnn_dim * 3,
                out_features=num_ner_tag,
                dropout=logit_drop,
                hidden_ratio=1.5,
            )
        elif self.head_type == 'relation_aware':
            self.score_head = RelationAwareSpanScoreHead(
                in_features=cnn_dim * 3,
                out_features=num_ner_tag,
                relation_dim=relation_dim,
                dropout=logit_drop,
            )
        elif self.head_type == 'length_gated':
            self.score_head = LengthGatedSpanScoreHead(
                branch_dim=cnn_dim,
                out_features=num_ner_tag,
                length_bins=length_gate_bins,
                dropout=logit_drop,
            )
        else:
            self.score_head = nn.Linear(cnn_dim * 3, num_ner_tag)
            torch.nn.init.xavier_normal_(self.score_head.weight.data)
        # Boundary-span collaborative supervision (training only).  The two
        # heads reuse the start/end representations that already form the
        # span map, so they reinforce boundary evidence without adding any
        # inference-time branch or replacing SNSA/HSR.
        if self.boundary_aux_weight > 0 or self.boundary_align_weight > 0:
            self.start_boundary_head = nn.Linear(biaffine_size, num_ner_tag)
            self.end_boundary_head = nn.Linear(biaffine_size, num_ner_tag)
            torch.nn.init.xavier_normal_(self.start_boundary_head.weight.data)
            torch.nn.init.xavier_normal_(self.end_boundary_head.weight.data)
            torch.nn.init.zeros_(self.start_boundary_head.bias.data)
            torch.nn.init.zeros_(self.end_boundary_head.bias.data)
        if self.conflict_rank_weight > 0:
            self.register_buffer('conflict_train_steps', torch.zeros((), dtype=torch.long))
        # A training-only IoU-quality head receives the HSR branch directly.
        # It learns whether a candidate is a complete entity span rather than
        # merely a high-scoring, partially overlapping fragment.
        if self.quality_aux_weight > 0:
            quality_hidden = max(64, cnn_dim // 2)
            self.span_quality_head = nn.Sequential(
                nn.LayerNorm(cnn_dim),
                nn.Linear(cnn_dim, quality_hidden),
                nn.GELU(),
                nn.Dropout(logit_drop),
                nn.Linear(quality_hidden, 1),
            )
            for module in self.span_quality_head:
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_normal_(module.weight.data)
                    torch.nn.init.zeros_(module.bias.data)
        # Training-only hierarchy supervision for HSR. It predicts two
        # non-exclusive roles for a gold span: contains-an-entity and
        # is-contained-by-an-entity. This retains the rare "middle" role
        # without making it an unstable standalone four-class category.
        if self.hierarchy_aux_weight > 0:
            hierarchy_hidden = max(64, cnn_dim // 2)
            self.hierarchy_role_head = nn.Sequential(
                nn.LayerNorm(cnn_dim),
                nn.Linear(cnn_dim, hierarchy_hidden),
                nn.GELU(),
                nn.Dropout(logit_drop),
                nn.Linear(hierarchy_hidden, 2),
            )
            for module in self.hierarchy_role_head:
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_normal_(module.weight.data)
                    torch.nn.init.zeros_(module.bias.data)
        if self.difficulty_curriculum:
            self.register_buffer('curriculum_train_steps', torch.zeros((), dtype=torch.long))
        # BHPC: type prototypes regularize positive spans while boundary-neighbour
        # non-entities are used as hard negatives. Buffers make it harmless when
        # BHPC is disabled and keep prototype updates out of the optimizer.
        if self.bhpc_weight > 0:
            self.bhpc_projection = nn.Sequential(
                nn.LayerNorm(cnn_dim * 3),
                nn.Linear(cnn_dim * 3, bhpc_dim, bias=False),
            )
            self.register_buffer('bhpc_prototypes', torch.zeros(num_ner_tag, bhpc_dim))
            self.register_buffer('bhpc_seen', torch.zeros(num_ner_tag, dtype=torch.bool))
            if bhpc_label_weights is None:
                bhpc_label_weights = torch.ones(num_ner_tag, dtype=torch.float32)
            else:
                bhpc_label_weights = torch.as_tensor(bhpc_label_weights, dtype=torch.float32)
                if bhpc_label_weights.numel() != num_ner_tag:
                    raise ValueError(
                        f"bhpc_label_weights must have {num_ner_tag} values, "
                        f"got {bhpc_label_weights.numel()}"
                    )
            self.register_buffer('bhpc_label_weights', bhpc_label_weights.view(-1))
            self.register_buffer('bhpc_train_steps', torch.zeros((), dtype=torch.long))
        if self.use_length_bias:
            self.length_bias = nn.Embedding(length_bias_bins, num_ner_tag)
            torch.nn.init.zeros_(self.length_bias.weight.data)
            row_ids = torch.arange(512).view(512, 1)
            col_ids = torch.arange(512).view(1, 512)
            length_ids = (row_ids - col_ids).abs().clamp(max=length_bias_bins - 1)
            self.register_buffer('length_bias_ids', length_ids.long())
        self.logit_drop = logit_drop

    def _asl_loss_matrix(self, logits, targets, valid_mask):
        prob = torch.sigmoid(logits)
        positive = targets.gt(0.5) & valid_mask
        negative = targets.le(0.5) & valid_mask
        loss = logits.new_zeros(logits.shape)

        neg_prob = (1.0 - prob).clamp(min=1e-6, max=1.0 - 1e-6)
        if self.asl_clip > 0:
            neg_prob = (neg_prob + self.asl_clip).clamp(min=1e-6, max=1.0 - 1e-6)
        neg_loss = -torch.log(neg_prob)
        if self.asl_gamma_neg > 0:
            neg_loss = neg_loss * torch.pow((1.0 - neg_prob).clamp(min=1e-6), self.asl_gamma_neg)
        loss = torch.where(negative, neg_loss, loss)

        pos_prob = prob.clamp(min=1e-6, max=1.0 - 1e-6)
        pos_loss = -torch.log(pos_prob) * self.label_loss_weights.to(logits.dtype)
        if self.asl_gamma_pos > 0:
            pos_loss = pos_loss * torch.pow((1.0 - pos_prob).clamp(min=1e-6), self.asl_gamma_pos)
        loss = torch.where(positive, pos_loss, loss)
        return loss, positive, negative

    def _difficulty_curriculum_weights(self, matrix, progress):
        """Return per-span/class weights for the original classification loss.

        The curriculum is deliberately not an auxiliary objective: it only
        reallocates main-task learning pressure from easy independent spans to
        nested/long positives and one-boundary-off negative spans.  The weight
        starts at one everywhere and transitions smoothly after warm-up.
        """
        weights = matrix.new_ones(matrix.shape)
        if progress <= 0:
            return weights
        valid_by_label = matrix.ne(-100)
        targets = matrix.masked_fill(~valid_by_label, 0.0).gt(0.5)
        batch_size, length, _, _ = targets.shape
        device = matrix.device
        upper = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool))
        gold_spans = targets.any(dim=-1) & upper.unsqueeze(0)
        nested_spans = torch.zeros_like(gold_spans)
        near_boundary = torch.zeros_like(gold_spans)
        for batch_id in range(batch_size):
            gold_indices = gold_spans[batch_id].nonzero(as_tuple=False)
            if gold_indices.numel() == 0:
                continue
            starts = gold_indices[:, 0]
            ends = gold_indices[:, 1]
            contains_pair = (
                (starts[:, None] <= starts[None, :])
                & (ends[None, :] <= ends[:, None])
                & ((starts[:, None] != starts[None, :]) | (ends[:, None] != ends[None, :]))
            )
            nested = contains_pair.any(dim=0) | contains_pair.any(dim=1)
            nested_spans[batch_id, starts[nested], ends[nested]] = True
            # A one-token endpoint expansion/shrink is a frequent NER error;
            # it is safe only when the resulting span is not another gold span.
            for start, end in gold_indices.tolist():
                for next_start, next_end in ((start - 1, end), (start + 1, end),
                                              (start, end - 1), (start, end + 1)):
                    if 0 <= next_start <= next_end < length:
                        near_boundary[batch_id, next_start, next_end] = True
        near_boundary &= ~gold_spans
        row_ids = torch.arange(length, device=device).view(length, 1)
        col_ids = torch.arange(length, device=device).view(1, length)
        long_spans = (col_ids - row_ids + 1 >= self.curriculum_long_span) & upper
        positive_scale = 1.0 + float(progress) * (
            self.curriculum_nested_boost * nested_spans.to(matrix.dtype)
            + self.curriculum_long_boost * long_spans.unsqueeze(0).to(matrix.dtype)
        )
        negative_scale = 1.0 + float(progress) * self.curriculum_hard_neg_boost * near_boundary.to(matrix.dtype)
        weights = torch.where(
            targets,
            positive_scale.unsqueeze(-1).expand_as(weights),
            negative_scale.unsqueeze(-1).expand_as(weights),
        )
        return weights

    def _span_loss(self, final_score, matrix, batch_size, curriculum_progress=0.0):
        valid_mask = matrix.ne(-100)
        targets = matrix.masked_fill(~valid_mask, 0.0).float()

        if self.loss_type in ('asl', 'balanced_asl'):
            loss, positive, negative = self._asl_loss_matrix(final_score.float(), targets, valid_mask)
        else:
            loss = F.binary_cross_entropy_with_logits(final_score, targets, reduction='none')
            positive = targets.gt(0.5) & valid_mask
            negative = targets.le(0.5) & valid_mask

        if curriculum_progress > 0:
            loss = loss * self._difficulty_curriculum_weights(matrix, curriculum_progress)

        if self.loss_type == 'balanced_asl':
            flat_loss = loss.view(batch_size, -1)
            flat_positive = positive.view(batch_size, -1).float()
            flat_negative = negative.view(batch_size, -1).float()
            pos_loss = (flat_loss * flat_positive).sum(dim=-1) / flat_positive.sum(dim=-1).clamp_min(1.0)
            neg_loss = (flat_loss * flat_negative).sum(dim=-1) / flat_negative.sum(dim=-1).clamp_min(1.0)
            return (pos_loss + self.balanced_neg_weight * neg_loss).mean()

        sample_mask = valid_mask.float().view(batch_size, -1)
        return ((loss.view(batch_size, -1) * sample_mask).sum(dim=-1)).mean()

    def _boundary_auxiliary_loss(self, start_logits, end_logits, final_score, matrix):
        """Boundary supervision and positive-span alignment.

        Gold span labels are projected to their start and end tokens.  The
        alignment term is evaluated only on gold entity spans, preventing the
        overwhelming number of non-entity spans from turning it into a trivial
        all-negative objective.
        """
        valid_span = matrix.ne(-100)
        targets = matrix.masked_fill(~valid_span, 0.0).gt(0.5)
        valid_tokens = valid_span.any(dim=2).any(dim=-1)
        start_targets = targets.any(dim=2)
        end_targets = targets.any(dim=1)

        def boundary_loss(logits, boundary_targets):
            valid = valid_tokens.unsqueeze(-1)
            positive = boundary_targets & valid
            negative = (~boundary_targets) & valid
            prob = torch.sigmoid(logits)
            neg_prob = (1.0 - prob).clamp(min=1e-6, max=1.0 - 1e-6)
            if self.asl_clip > 0:
                neg_prob = (neg_prob + self.asl_clip).clamp(min=1e-6, max=1.0 - 1e-6)
            neg_loss = -torch.log(neg_prob)
            if self.asl_gamma_neg > 0:
                neg_loss = neg_loss * torch.pow(
                    (1.0 - neg_prob).clamp(min=1e-6), self.asl_gamma_neg
                )
            pos_prob = prob.clamp(min=1e-6, max=1.0 - 1e-6)
            pos_loss = -torch.log(pos_prob) * self.label_loss_weights.view(1, 1, -1)
            if self.asl_gamma_pos > 0:
                pos_loss = pos_loss * torch.pow(
                    (1.0 - pos_prob).clamp(min=1e-6), self.asl_gamma_pos
                )
            # Balance positive and negative token decisions.  Without this,
            # the auxiliary branch is dominated by non-boundary tokens.
            pos_mean = (pos_loss * positive).sum() / positive.sum().clamp_min(1)
            neg_mean = (neg_loss * negative).sum() / negative.sum().clamp_min(1)
            return pos_mean + neg_mean

        aux_loss = boundary_loss(start_logits, start_targets) + boundary_loss(end_logits, end_targets)
        positive_spans = targets & torch.triu(
            torch.ones(targets.size(1), targets.size(2), device=targets.device, dtype=torch.bool)
        ).view(1, targets.size(1), targets.size(2), 1)
        if positive_spans.any():
            start_prob = torch.sigmoid(start_logits).unsqueeze(2)
            end_prob = torch.sigmoid(end_logits).unsqueeze(1)
            # Geometric mean treats the two boundaries symmetrically and keeps
            # the auxiliary confidence on the same probability scale as span
            # classification.
            pair_prob = torch.sqrt((start_prob * end_prob).clamp_min(1e-6))
            align_loss = F.mse_loss(torch.sigmoid(final_score)[positive_spans], pair_prob[positive_spans])
        else:
            align_loss = final_score.new_zeros(())
        return aux_loss, align_loss

    def _conflict_ranking_loss(self, final_score, matrix):
        """Rank gold spans above incompatible, non-nested hard negatives.

        The decoder permits containment but discards crossing spans.  This
        training-only objective therefore compares each gold span only with
        non-entity candidates that overlap it without either span containing
        the other.  Gold nested spans are deliberately excluded from the
        negative set.
        """
        valid_by_label = matrix.ne(-100)
        targets = matrix.masked_fill(~valid_by_label, 0.0).gt(0.5)
        batch_size, length, _, _ = targets.shape
        device = targets.device
        upper = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool))
        valid_spans = valid_by_label.any(dim=-1) & upper.unsqueeze(0)
        gold_spans = targets.any(dim=-1) & upper.unsqueeze(0)
        candidate_spans = valid_spans & ~gold_spans
        max_candidate_logits = final_score.max(dim=-1).values

        starts = torch.arange(length, device=device).view(length, 1)
        ends = torch.arange(length, device=device).view(1, length)
        losses = []
        # Gold spans are sparse, so iterating over them avoids allocating an
        # impractical [L, L, L, L] pairwise conflict tensor.
        for batch_id, start, end, label in (targets & upper.view(1, length, length, 1)).nonzero(
            as_tuple=False
        ).tolist():
            overlap = (starts <= end) & (ends >= start)
            nested = ((starts >= start) & (ends <= end)) | ((starts <= start) & (ends >= end))
            conflict = candidate_spans[batch_id] & overlap & ~nested
            if conflict.any():
                hardest_conflict = max_candidate_logits[batch_id].masked_fill(~conflict, -torch.inf).max()
                gold_logit = final_score[batch_id, start, end, label]
                losses.append(F.relu(self.conflict_margin - gold_logit + hardest_conflict))
        if not losses:
            return final_score.new_zeros(())
        return torch.stack(losses).mean()

    def _span_quality_loss(self, quality_logits, matrix):
        """Supervise HSR features with a class-agnostic span IoU target.

        A partially overlapping candidate receives its maximum IoU with any
        gold entity in the same sentence.  Exact entities get target 1, while
        unrelated candidates get 0.  Positive/negative terms are averaged
        separately so the enormous background span set cannot dominate the
        auxiliary objective.
        """
        valid_by_label = matrix.ne(-100)
        targets = matrix.masked_fill(~valid_by_label, 0.0).gt(0.5)
        batch_size, length, _, _ = targets.shape
        device = targets.device
        upper = torch.triu(torch.ones(length, length, device=device, dtype=torch.bool))
        valid_spans = valid_by_label.any(dim=-1) & upper.unsqueeze(0)
        gold_spans = targets.any(dim=-1) & upper.unsqueeze(0)

        row_ids = torch.arange(length, device=device).view(length, 1, 1)
        col_ids = torch.arange(length, device=device).view(1, length, 1)
        candidate_length = (col_ids - row_ids + 1).clamp_min(1).float()
        quality_target = quality_logits.new_zeros(batch_size, length, length)
        for batch_id in range(batch_size):
            gold_indices = gold_spans[batch_id].nonzero(as_tuple=False)
            if gold_indices.numel() == 0:
                continue
            gold_starts = gold_indices[:, 0].view(1, 1, -1)
            gold_ends = gold_indices[:, 1].view(1, 1, -1)
            intersection = (
                torch.minimum(col_ids, gold_ends) - torch.maximum(row_ids, gold_starts) + 1
            ).clamp_min(0).float()
            gold_length = (gold_ends - gold_starts + 1).float()
            union = candidate_length + gold_length - intersection
            quality_target[batch_id] = (intersection / union.clamp_min(1.0)).max(dim=-1).values

        logits = quality_logits.squeeze(-1)
        loss_matrix = F.binary_cross_entropy_with_logits(logits, quality_target, reduction='none')
        near_gold = valid_spans & quality_target.ge(self.quality_min_iou)
        background = valid_spans & ~near_gold
        near_loss = (loss_matrix * near_gold).sum() / near_gold.sum().clamp_min(1)
        background_loss = (loss_matrix * background).sum() / background.sum().clamp_min(1)
        return near_loss + background_loss

    def _hierarchy_role_loss(self, role_logits, matrix):
        """Teach HSR whether each gold span contains or is contained.

        Only gold spans are used. For both attributes, positive and negative
        cases are averaged separately, which prevents independent entities from
        overwhelming the much rarer nested roles in GENIA.
        """
        valid_by_label = matrix.ne(-100)
        targets = matrix.masked_fill(~valid_by_label, 0.0).gt(0.5)
        batch_size, length, _, _ = targets.shape
        upper = torch.triu(torch.ones(length, length, device=targets.device, dtype=torch.bool))
        gold_spans = targets.any(dim=-1) & upper.unsqueeze(0)
        all_logits = []
        all_targets = []
        for batch_id in range(batch_size):
            gold_indices = gold_spans[batch_id].nonzero(as_tuple=False)
            if gold_indices.numel() == 0:
                continue
            starts = gold_indices[:, 0]
            ends = gold_indices[:, 1]
            # pair[i, j] means span i strictly contains span j.
            contains_pair = (
                (starts[:, None] <= starts[None, :])
                & (ends[None, :] <= ends[:, None])
                & ((starts[:, None] != starts[None, :]) | (ends[:, None] != ends[None, :]))
            )
            role_targets = torch.stack(
                [contains_pair.any(dim=1), contains_pair.any(dim=0)], dim=-1
            ).to(role_logits.dtype)
            all_logits.append(role_logits[batch_id, starts, ends])
            all_targets.append(role_targets)
        if not all_logits:
            return role_logits.new_zeros(())
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        loss_matrix = F.binary_cross_entropy_with_logits(all_logits, all_targets, reduction='none')
        losses = []
        # Balance contains and contained independently across the minibatch.
        for role_idx in range(2):
            positives = all_targets[:, role_idx].gt(0.5)
            negatives = ~positives
            if positives.any() and negatives.any():
                losses.append(loss_matrix[positives, role_idx].mean())
                losses.append(loss_matrix[negatives, role_idx].mean())
        if not losses:
            return role_logits.new_zeros(())
        return torch.stack(losses).mean()

    def _bhpc_loss(self, span_features, final_score, matrix):
        """Boundary-aware hard-negative prototype contrastive loss.

        Each gold span is aligned with an EMA class prototype. Its four immediate
        boundary neighbours are candidates for hard negatives; gold spans of any
        type (including nested spans) are never treated as negatives.
        """
        if self.bhpc_weight <= 0:
            return final_score.new_zeros(())

        valid_mask = matrix.ne(-100)
        targets = matrix.masked_fill(~valid_mask, 0.0).gt(0.5)
        # The lower triangle is not a distinct text span. Keeping only the upper
        # triangle also avoids duplicated supervision if the input matrix is
        # symmetric.
        length = targets.size(1)
        upper = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=targets.device)
        ).view(1, length, length, 1)
        positives = (targets & upper).nonzero(as_tuple=False)
        if positives.numel() == 0:
            return final_score.new_zeros(())

        batch_ids, starts, ends, labels = positives.unbind(dim=1)
        positive_features = span_features[batch_ids, starts, ends]
        # Under fp16, the default normalize epsilon (1e-12) underflows to zero.
        # That can turn an all-zero/near-zero projection into NaN and poison the
        # whole optimizer state. Keep the contrastive branch and its EMA state in
        # fp32; its contribution remains differentiable to the main network.
        device_type = span_features.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            positive_embeddings = F.normalize(
                self.bhpc_projection(positive_features.float()), dim=-1, eps=1e-6
            )

        seen_before = self.bhpc_seen.detach().clone()
        available = seen_before[labels]
        prototype_loss = final_score.new_zeros(())
        if available.any():
            prototype_embeddings = F.normalize(self.bhpc_prototypes, dim=-1)
            logits = positive_embeddings[available] @ prototype_embeddings.t()
            logits = logits / self.bhpc_temperature
            logits[:, ~seen_before] = torch.finfo(logits.dtype).min
            prototype_losses = F.cross_entropy(logits, labels[available], reduction='none')
            prototype_loss = self.bhpc_prototype_scale * (
                prototype_losses * self.bhpc_label_weights[labels[available]]
            ).mean()

        # Pick the most confident adjacent non-entity span for each gold span.
        # This concentrates the auxiliary signal on one-token boundary errors.
        negative_features = []
        negative_anchor_features = []
        negative_labels = []
        offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
        for batch_id, start, end, label in positives.tolist():
            candidates = []
            for delta_start, delta_end in offsets:
                neg_start = start + delta_start
                neg_end = end + delta_end
                if neg_start < 0 or neg_end < neg_start or neg_end >= length:
                    continue
                # A valid entity of another type may share or almost share a
                # boundary with this span; it must remain excluded.
                if not valid_mask[batch_id, neg_start, neg_end, label]:
                    continue
                if targets[batch_id, neg_start, neg_end].any():
                    continue
                score = final_score[batch_id, neg_start, neg_end, label].detach()
                candidates.append((score, neg_start, neg_end))
            if candidates:
                _, neg_start, neg_end = max(candidates, key=lambda item: item[0].item())
                negative_features.append(span_features[batch_id, neg_start, neg_end])
                negative_anchor_features.append(span_features[batch_id, start, end])
                negative_labels.append(label)

        boundary_loss = final_score.new_zeros(())
        if negative_features:
            negative_features = torch.stack(negative_features, dim=0)
            with torch.autocast(device_type=device_type, enabled=False):
                negative_embeddings = F.normalize(
                    self.bhpc_projection(negative_features.float()), dim=-1, eps=1e-6
                )
            negative_anchor_features = torch.stack(negative_anchor_features, dim=0)
            negative_labels = torch.tensor(negative_labels, device=labels.device, dtype=labels.dtype)
            valid_negatives = seen_before[negative_labels]
            if valid_negatives.any():
                prototypes = F.normalize(self.bhpc_prototypes[negative_labels[valid_negatives]], dim=-1)
                with torch.autocast(device_type=device_type, enabled=False):
                    positive_by_span = F.normalize(
                        self.bhpc_projection(negative_anchor_features[valid_negatives].float()),
                        dim=-1,
                        eps=1e-6,
                    )
                pos_similarity = (positive_by_span * prototypes).sum(dim=-1)
                neg_similarity = (negative_embeddings[valid_negatives] * prototypes).sum(dim=-1)
                boundary_losses = F.relu(self.bhpc_margin - pos_similarity + neg_similarity)
                boundary_loss = self.bhpc_boundary_scale * (
                    boundary_losses * self.bhpc_label_weights[negative_labels[valid_negatives]]
                ).mean()

        # Update class prototypes after evaluating the current batch so anchors
        # cannot trivially match a prototype that was just built from themselves.
        with torch.no_grad():
            for label in labels.unique():
                label_id = int(label.item())
                class_mean = positive_embeddings[labels == label].mean(dim=0)
                if self.bhpc_seen[label_id]:
                    self.bhpc_prototypes[label_id].mul_(self.bhpc_momentum).add_(
                        class_mean * (1.0 - self.bhpc_momentum)
                    )
                else:
                    self.bhpc_prototypes[label_id].copy_(class_mean)
                    self.bhpc_seen[label_id] = True

        return prototype_loss + boundary_loss

    def forward(self, input_ids, bpe_len, indexes, matrix,raw_words):

        attention_mask = seq_len_to_mask(bpe_len)  # bsz x length x length
        outputs = self.pretrain_model(input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_states = outputs['last_hidden_state']
        scat_max  = scatter_max(last_hidden_states, index=indexes, dim=1)[0]  # bsz x word_len x hidden_size
        state = scat_max[:,1:]
        lengths, _ = indexes.max(dim=-1)
        head_state = self.head_mlp(state)
        tail_state = self.tail_mlp(state)
        # Keep the unaugmented endpoint representations for the optional
        # training-only boundary heads.  The copies below are subsequently
        # extended with a biaffine bias coordinate.
        boundary_head_state = head_state
        boundary_tail_state = tail_state
        if hasattr(self, 'U'):
            scores1 = torch.einsum('bxi, oij, byj -> boxy', head_state, self.U, tail_state)
        else:
            scores1 = self.biaffine(head_state, tail_state)
        head_state = torch.cat([head_state, torch.ones_like(head_state[..., :1])], dim=-1)
        tail_state = torch.cat([tail_state, torch.ones_like(tail_state[..., :1])], dim=-1)
        affined_cat = torch.cat([head_state.unsqueeze(2).expand(-1, -1, tail_state.size(1), -1),
                                 tail_state.unsqueeze(1).expand(-1, head_state.size(1), -1, -1)], dim=-1)
        mask = seq_len_to_mask(lengths)  # bsz x length x length
        mask = mask[:, None] * mask.unsqueeze(-1)
        pad_mask = mask[:, None].eq(0)
        # pad_mask1 = pad_mask
        pad_mask1 = pad_mask * torch.tril(pad_mask).ne(0)
        if hasattr(self, 'size_embedding'):
            size_embedded = self.size_embedding(self.span_size_ids[:state.size(1), :state.size(1)])
            affined_cat = torch.cat(
                [self.dropout(affined_cat), self.dropout(size_embedded).unsqueeze(0).expand(state.size(0), -1, -1, -1)],
                dim=-1)

        scores2 = torch.einsum('bmnh,kh->bkmn', affined_cat, self.W)  # bsz x dim x L x L
        scores = scores2 + scores1  # bsz x dim x L x L 
        
        if hasattr(self, 'cnn1'):
            if self.logit_drop != 0:
                scores = F.dropout(scores, p=self.logit_drop, training=self.training)
            u_scores1 = scores.masked_fill(pad_mask1, 0)
            u_score1= self.cnn1(u_scores1, pad_mask1,self.training)

        u_score = torch.concat([scores, u_score1],dim=1)
        span_features = u_score.permute(0, 2, 3, 1)
        if self.head_type == 'relation_aware':
            final_score = self.score_head(span_features, lengths)
        else:
            final_score = self.score_head(span_features)
        if self.use_length_bias:
            length_bias = self.length_bias(self.length_bias_ids[:state.size(1), :state.size(1)])
            final_score = final_score + length_bias.unsqueeze(0)
        assert final_score.size(-1) == matrix.size(-1)
        if self.training:
            if self.difficulty_curriculum:
                self.curriculum_train_steps.add_(1)
                elapsed = self.curriculum_train_steps.item() - self.curriculum_warmup_steps
                curriculum_progress = min(1.0, max(0.0, elapsed / self.curriculum_ramp_steps))
            else:
                curriculum_progress = 0.0
            span_loss = self._span_loss(
                final_score, matrix, input_ids.size(0), curriculum_progress=curriculum_progress
            )
            if hasattr(self, 'start_boundary_head'):
                start_logits = self.start_boundary_head(boundary_head_state)
                end_logits = self.end_boundary_head(boundary_tail_state)
                boundary_aux_loss, boundary_align_loss = self._boundary_auxiliary_loss(
                    start_logits, end_logits, final_score, matrix
                )
            else:
                boundary_aux_loss = span_loss.new_zeros(())
                boundary_align_loss = span_loss.new_zeros(())
            if self.bhpc_weight > 0:
                self.bhpc_train_steps.add_(1)
                if self.bhpc_train_steps.item() > self.bhpc_warmup_steps:
                    bhpc_loss = self._bhpc_loss(span_features, final_score, matrix)
                else:
                    bhpc_loss = span_loss.new_zeros(())
            else:
                bhpc_loss = span_loss.new_zeros(())
            if self.conflict_rank_weight > 0:
                self.conflict_train_steps.add_(1)
                if self.conflict_train_steps.item() > self.conflict_warmup_steps:
                    conflict_rank_loss = self._conflict_ranking_loss(final_score, matrix)
                else:
                    conflict_rank_loss = span_loss.new_zeros(())
            else:
                conflict_rank_loss = span_loss.new_zeros(())
            if hasattr(self, 'span_quality_head'):
                # span_features is [raw span | SNSA | HSR]; supervise only the
                # hierarchical branch so the auxiliary task complements HSR.
                quality_logits = self.span_quality_head(span_features[..., self.cnn_dim * 2:])
                quality_loss = self._span_quality_loss(quality_logits, matrix)
            else:
                quality_loss = span_loss.new_zeros(())
            if hasattr(self, 'hierarchy_role_head'):
                hierarchy_logits = self.hierarchy_role_head(span_features[..., self.cnn_dim * 2:])
                hierarchy_loss = self._hierarchy_role_loss(hierarchy_logits, matrix)
            else:
                hierarchy_loss = span_loss.new_zeros(())
            loss = (
                span_loss + self.bhpc_weight * bhpc_loss
                + self.boundary_aux_weight * boundary_aux_loss
                + self.boundary_align_weight * boundary_align_loss
                + self.conflict_rank_weight * conflict_rank_loss
                + self.quality_aux_weight * quality_loss
                + self.hierarchy_aux_weight * hierarchy_loss
            )
            return {
                'loss': loss,
                'span_loss': span_loss.detach(),
                'bhpc_loss': bhpc_loss.detach(),
                'boundary_aux_loss': boundary_aux_loss.detach(),
                'boundary_align_loss': boundary_align_loss.detach(),
                'conflict_rank_loss': conflict_rank_loss.detach(),
                'quality_loss': quality_loss.detach(),
                'hierarchy_loss': hierarchy_loss.detach(),
                'curriculum_progress': span_loss.new_tensor(curriculum_progress),
            }
        return {'scores': final_score}
