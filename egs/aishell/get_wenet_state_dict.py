import collections

import torch


key_mapping = {
    "encoder.global_cmvn.mean": "encoder.global_cmvn.mean",
    "encoder.global_cmvn.istd": "encoder.global_cmvn.istd",
    "encoder.embed.conv.0.weight": "encoder.subsampling.conv.0.weight",
    "encoder.embed.conv.0.bias": "encoder.subsampling.conv.0.bias",
    "encoder.embed.conv.2.weight": "encoder.subsampling.conv.2.weight",
    "encoder.embed.conv.2.bias": "encoder.subsampling.conv.2.bias",
    "encoder.embed.out.0.weight": "encoder.subsampling.linear.0.weight",
    "encoder.embed.out.0.bias": "encoder.subsampling.linear.0.bias",
    "encoder.after_norm.weight": None,
    "encoder.after_norm.bias": None,
    "encoder.encoders.0.self_attn.pos_bias_u": "encoder.encoders.0.attn._attn_module.u_bias",
    "encoder.encoders.0.self_attn.pos_bias_v": "encoder.encoders.0.attn._attn_module.v_bias",
    "encoder.encoders.0.self_attn.linear_q.weight": "encoder.encoders.0.attn._attn_module.linear_q.weight",
    "encoder.encoders.0.self_attn.linear_q.bias": None,
    "encoder.encoders.0.self_attn.linear_k.weight": "encoder.encoders.0.attn._attn_module.linear_k.weight",
    "encoder.encoders.0.self_attn.linear_k.bias": None,
    "encoder.encoders.0.self_attn.linear_v.weight": "encoder.encoders.0.attn._attn_module.linear_v.weight",
    "encoder.encoders.0.self_attn.linear_v.bias": None,
    "encoder.encoders.0.self_attn.linear_out.weight": "encoder.encoders.0.attn._attn_module.linear_out.weight",
    "encoder.encoders.0.self_attn.linear_out.bias": None,
    "encoder.encoders.0.self_attn.linear_pos.weight": "encoder.encoders.0.attn._attn_module.linear_pos.weight",
    "encoder.encoders.0.feed_forward.w_1.weight": "encoder.encoders.0.feed_forward2.layer.1.weight",
    "encoder.encoders.0.feed_forward.w_1.bias": "encoder.encoders.0.feed_forward2.layer.1.bias",
    "encoder.encoders.0.feed_forward.w_2.weight": "encoder.encoders.0.feed_forward2.layer.4.weight",
    "encoder.encoders.0.feed_forward.w_2.bias": "encoder.encoders.0.feed_forward2.layer.4.bias",
    "encoder.encoders.0.feed_forward_macaron.w_1.weight": "encoder.encoders.0.feed_forward1.layer.1.weight",
    "encoder.encoders.0.feed_forward_macaron.w_1.bias": "encoder.encoders.0.feed_forward1.layer.1.bias",
    "encoder.encoders.0.feed_forward_macaron.w_2.weight": "encoder.encoders.0.feed_forward1.layer.4.weight",
    "encoder.encoders.0.feed_forward_macaron.w_2.bias": "encoder.encoders.0.feed_forward1.layer.4.bias",
    "encoder.encoders.0.conv_module.pointwise_conv1.weight": "encoder.encoders.0.conv.pointwise_conv1.weight",
    "encoder.encoders.0.conv_module.pointwise_conv1.bias": "encoder.encoders.0.conv.pointwise_conv1.bias",
    "encoder.encoders.0.conv_module.depthwise_conv.weight": "encoder.encoders.0.conv.depthwise_conv.conv.weight",
    "encoder.encoders.0.conv_module.depthwise_conv.bias": "encoder.encoders.0.conv.depthwise_conv.conv.bias",
    "encoder.encoders.0.conv_module.norm.weight": "encoder.encoders.0.conv.batch_norm.weight",
    "encoder.encoders.0.conv_module.norm.bias": "encoder.encoders.0.conv.batch_norm.bias",
    "encoder.encoders.0.conv_module.norm.running_mean": "encoder.encoders.0.conv.batch_norm.running_mean",
    "encoder.encoders.0.conv_module.norm.running_var": "encoder.encoders.0.conv.batch_norm.running_var",
    "encoder.encoders.0.conv_module.norm.num_batches_tracked": "encoder.encoders.0.conv.batch_norm.num_batches_tracked",
    "encoder.encoders.0.conv_module.pointwise_conv2.weight": "encoder.encoders.0.conv.pointwise_conv2.weight",
    "encoder.encoders.0.conv_module.pointwise_conv2.bias": "encoder.encoders.0.conv.pointwise_conv2.bias",
    "encoder.encoders.0.norm_ff.weight": "encoder.encoders.0.feed_forward2.layer.0.weight",
    "encoder.encoders.0.norm_ff.bias": "encoder.encoders.0.feed_forward2.layer.0.bias",
    "encoder.encoders.0.norm_mha.weight": None,
    "encoder.encoders.0.norm_mha.bias": None,
    "encoder.encoders.0.norm_ff_macaron.weight": "encoder.encoders.0.feed_forward1.layer.0.weight",
    "encoder.encoders.0.norm_ff_macaron.bias": "encoder.encoders.0.feed_forward1.layer.0.bias",
    "encoder.encoders.0.norm_conv.weight": "encoder.encoders.0.conv.layer_norm.weight",
    "encoder.encoders.0.norm_conv.bias": "encoder.encoders.0.conv.layer_norm.bias",
    "encoder.encoders.0.norm_final.weight": "encoder.encoders.0.layer_norm.weight",
    "encoder.encoders.0.norm_final.bias": "encoder.encoders.0.layer_norm.bias",
    "encoder.encoders.0.concat_linear.weight": None,
    "encoder.encoders.0.concat_linear.bias": None,
    "decoder.embed.0.weight": None,
    "decoder.after_norm.weight": None,
    "decoder.after_norm.bias": None,
    "decoder.output_layer.weight": None,
    "decoder.output_layer.bias": None,
    "ctc.ctc_lo.weight": "ctc_decoder.linear.weight",
    "ctc.ctc_lo.bias": "ctc_decoder.linear.bias",
}


def lookup_key(key: str) -> str:
    if key.startswith("encoder.encoders."):
        parts = key.split(".", 3)
        original_idx = parts[2]
        parts[2] = "0"
        query_key = ".".join(parts)
        value = key_mapping.get(query_key)
        if value is not None:
            value_parts = value.split(".", 3)
            value_parts[2] = original_idx
            value = ".".join(value_parts)
    else:
        value = key_mapping.get(key)
    return value


def main():
    path = "model_params_from_wenet/20210204_conformer_exp/final.pt"
    res: collections.OrderedDict = torch.load(path, map_location="cpu")
    state_dict = collections.OrderedDict()
    for k, v in res.items():
        key = lookup_key(k)
        if key is not None:
            state_dict[key] = v
            print(k, "=", key)
    torch.save(state_dict, "exp/state_dict_from_wenet.pt")


if __name__ == "__main__":
    main()
