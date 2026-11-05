"""
Decode a PerQ bitstream to a PNG image:

    python decompress.py --model-cfg cfgs/model/dcvc-rt_perq/dcvc-rt_intra_perq.yaml \\
        --weights <checkpoint> --input kodim01.bin --output kodim01_dec.png [--seed 0]

The compensator samples Gaussian noise, so the output depends on --seed.
"""
from perq.engine import *
from perq.io_utils import compute_scale_shift
from perq.bitstream import read_bitstream
from compress import build_model


def _parse_args():
    parser = argparse.ArgumentParser(description='PerQ: decompress one image', allow_abbrev=False)
    parser.add_argument('--model-cfg', type=str, required=True, help='model config (the one used to compress)')
    parser.add_argument('--weights', type=str, required=True, help='checkpoint (.pth / pytorch_model.bin)')
    parser.add_argument('--input', type=str, required=True, help='input bitstream')
    parser.add_argument('--output', type=str, required=True, help='output PNG')
    parser.add_argument('--num-rp', type=int, default=64, help='number of rate points the model was trained with')
    parser.add_argument('--seed', type=int, default=0, help='seed of the compensator noise')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s')
    logger = logging.getLogger('perq')

    model = build_model(logger, args.model_cfg, args.weights, args.num_rp, args.device)
    bit_stream, header = read_bitstream(args.input)
    H, W = header['height'], header['width']
    padding, unpadding = compressai.ops.compute_padding(H, W, min_div=model.intra_model.PAD_STRIDE)
    sps = {'height': H + padding[2] + padding[3], 'width': W + padding[0] + padding[1], 'ec_part': 0,
           'rp': torch.tensor([header['rp']], device=args.device)}

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.inference_mode():
        x_hat = model.decompress(bit_stream, sps)['x_hat']
    x_hat = F.pad(x_hat, unpadding).clamp(0., 1.)
    # The PNG conversion of evaluation
    scale, shift = compute_scale_shift(3, 8, None, None)
    x_hat = (x_hat[0].cpu() - shift.view(3, 1, 1)) / scale.view(3, 1, 1)
    torchvision.io.write_png(x_hat.round().to(torch.uint8), args.output)
    logger.info(f'{args.input} -> {args.output}: {W}x{H}, rp {header["rp"]}, {len(bit_stream) * 8 / (H * W):.4f} bpp')


if __name__ == '__main__':
    main()