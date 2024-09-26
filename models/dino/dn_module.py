import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from util.misc import inverse_sigmoid

class DNEncoder(nn.Module):
    
    def __init__(self, num_queries, num_classes, hidden_dim, label_enc):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.label_enc = label_enc

    def __call__(self, dn_args, training):
        """
        MLX adaptation of the prepare_for_cdn function.
        """

        if training:
            targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
            # Double dn_number for positive and negative queries
            dn_number = dn_number * 2
            known = [mx.ones_like(t['labels']) for t in targets]
            batch_size = len(known)
            known_num = [mx.sum(k) for k in known]
            if int(max(known_num)) == 0:
                dn_number = 1
            else:
                if dn_number >= 100:
                    dn_number = dn_number // (int(max(known_num) * 2))
                elif dn_number < 1:
                    dn_number = 1
            if dn_number == 0:
                dn_number = 1
            unmask_bbox = unmask_label = mx.concatenate(known)
            labels = mx.concatenate([t['labels'] for t in targets])
            boxes = mx.concatenate([t['boxes'] for t in targets])
            batch_idx = mx.concatenate([mx.full(t['labels'].shape, i) for i, t in enumerate(targets)])
            known_indice = mx.arange(unmask_label.shape[0])
            known_indice = mx.tile(known_indice, (2 * dn_number,)).flatten()
            known_labels = mx.tile(labels, (2 * dn_number,1)).flatten()
            known_bid = mx.tile(batch_idx, (2 * dn_number,1)).flatten()
            known_bboxs = mx.tile(boxes, (2 * dn_number, 1))
            known_labels_expaned = mx.array(known_labels)
            known_bbox_expand = mx.array(known_bboxs)
            if label_noise_ratio > 0:
                p = mx.random.uniform(shape=known_labels_expaned.shape)
                new_label = mx.random.randint(0, self.num_classes, shape=known_labels_expaned.shape)
                known_labels_expaned = mx.where(p < (label_noise_ratio * 0.5), known_labels_expaned, new_label)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * 2 * dn_number)
            positive_idx = mx.arange(len(boxes)).astype(mx.int16)  # Equivalent to torch.tensor(range(len(boxes))).long()
            positive_idx = mx.tile(positive_idx[None, ...], (dn_number, 1))  # Equivalent to repeat(dn_number, 1)

            # Adjust based on dn_number
            positive_idx += (mx.arange(dn_number).astype(mx.int16) * len(boxes) * 2).reshape(-1, 1)  # Equivalent to torch addition

            # Flatten the positive_idx
            positive_idx = positive_idx.flatten()

            # Calculate negative_idx
            negative_idx = positive_idx + len(boxes)

            if box_noise_scale > 0:
                known_bbox_ = mx.zeros_like(known_bboxs)
                known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
                known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2
                diff = mx.zeros_like(known_bboxs)
                diff[:, :2] = known_bboxs[:, 2:] / 2
                diff[:, 2:] = known_bboxs[:, 2:] / 2
                sign_choices = mx.array([-1.0, 1.0])
                rand_sign = sign_choices[mx.random.categorical(mx.array([0.5, 0.5]), shape=known_bboxs.shape)]
                rand_part = mx.random.uniform(shape=known_bboxs.shape)
                rand_part[negative_idx] += 1.0
                known_bbox_ = mx.clip(known_bbox_ + rand_sign * rand_part * diff * box_noise_scale , 0.0, 1.0) 

                known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
                known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

            input_label_embed = self.label_enc(known_labels_expaned)
            input_bbox_embed = inverse_sigmoid(known_bbox_expand)
            padding_label = mx.zeros((pad_size, self.hidden_dim))
            padding_bbox = mx.zeros((pad_size, 4))

            input_query_label = mx.tile(padding_label, (batch_size, 1, 1))
            input_query_bbox = mx.tile(padding_bbox, (batch_size, 1, 1))
            if len(known_num):
                map_known_indice = mx.concatenate([mx.arange(num) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = mx.concatenate([map_known_indice + single_pad * i for i in range(2 * dn_number)])
            if len(known_bid):
                input_query_label[known_bid, map_known_indice] = input_label_embed
                input_query_bbox[known_bid, map_known_indice] =  input_bbox_embed

            tgt_size = pad_size + self.num_queries
            attn_mask = mx.ones((tgt_size, tgt_size))

            attn_mask[pad_size:, :pad_size] = True
            for i in range(dn_number):
                if i == 0:
                    attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                if i == dn_number - 1:
                    attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
                else:
                    attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                    attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

            dn_meta = {
                'pad_size': pad_size,
                'num_dn_group': dn_number,
            }

            dn_meta = {'pad_size': pad_size, 'num_dn_group': dn_number}
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None

        return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
    MLX adaptation of the dn_post_process function.
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]

        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta['output_known_lbs_bboxes'] = out

    return outputs_class, outputs_coord



def _set_aux_loss(outputs_class, outputs_coord):
    aux_outputs = []
    for cls, coord in zip(outputs_class[:-1], outputs_coord[:-1]):
        aux_outputs.append({'pred_logits': cls, 'pred_boxes': coord})
    return aux_outputs
