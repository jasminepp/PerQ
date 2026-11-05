"""
Compress one PNG image to a PerQ bitstream:

    python compress.py --model-cfg cfgs/model/dcvc-rt_perq/dcvc-rt_intra_perq.yaml \\
        --weights <checkpoint> --rp 31 --input kodim01.png --output kodim01.bin

--rp is a rate point in [0, num_rp), 0 the lowest rate.
"""
from perq.engine import *
from perq import models
from perq.io_utils import compute_scale_shift
from perq.bitstream import write_bitstream


def _parse_args():
    parser = argparse.ArgumentParser(description='PerQ: compress one image', allow_abbrev=False)
    parser.add_argument('--model-cfg', type=str, required=True, help='model config')
    parser.add_argument('--weights', type=str, required=True, help='checkpoint (.pth / pytorch_model.bin)')
    parser.add_argument('--input', type=str, required=True, help='input PNG')
    parser.add_argument('--output', type=str, required=True, help='output bitstream')
    parser.add_argument('--rp', type=int, required=True, help='rate point')
    parser.add_argument('--num-rp', type=int, default=64, help='number of rate points the model was trained with')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def build_model(logger, model_cfg_path, weights, num_rp, device):
    # Decoding in another process must recompute the entropy parameters bit-exactly
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_cfg = ModelConfig(**load_yaml_config(model_cfg_path))
    model = models.create_compress_model(logger, model_cfg, num_rates=num_rp)
    state_dict = torch.load(os.path.expanduser(weights), map_location='cpu')
    missing, unexpected = model.load_pretrain(state_dict)
    missing = [k for k in missing if not k.startswith('gan.') and k != 'intra_model.rp_to_qp']
    assert not missing and not unexpected, \
        f'checkpoint does not match the model: missing {missing[:5]}, unexpected {unexpected[:5]}'
    model = model.to(device).eval()
    model.load_cache()
    return model


def read_image(path, device):
    img = torchvision.io.decode_image(torchvision.io.read_file(path), mode=torchvision.io.ImageReadMode.RGB)
    # The dataset's normalisation, as in evaluation
    scale, shift = compute_scale_shift(3, 8, None, None)
    return (scale.view(3, 1, 1) * img.float() + shift.view(3, 1, 1))[None].to(device)


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s')
    logger = logging.getLogger('perq')
    assert 0 <= args.rp < args.num_rp, f'--rp must be in [0, {args.num_rp}), got {args.rp}'

    model = build_model(logger, args.model_cfg, args.weights, args.num_rp, args.device)
    x = read_image(args.input, args.device)
    _, _, H, W = x.shape
    padding, _ = compressai.ops.compute_padding(H, W, min_div=model.intra_model.PAD_STRIDE)
    x_padded = F.pad(x, padding, mode='replicate')

    with torch.inference_mode():
        out = model.compress(x_padded, torch.tensor([args.rp], device=args.device))
    write_bitstream(args.output, out['bit_stream'], H, W, args.rp)
    bits = len(out['bit_stream']) * 8
    logger.info(f'{args.input} -> {args.output}: {bits} bits, {bits / (H * W):.4f} bpp (rp {args.rp})')


if __name__ == '__main__':
    main()