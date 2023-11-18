from comer.model.comer import CoMER
import yaml
import argparse
import torch

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    params = yaml.full_load(open(args.config, 'r'))

    model = CoMER(
        **params['model']
    )
    print(model)
    img = torch.randn(2, 1, 64, 100)
    img_mask = torch.zeros_like(img, dtype=torch.bool)[:, 0, :, :]
    tgt = torch.ones((2, 10), dtype=torch.long)
    print(model(img, img_mask, tgt))
    
    max_len = 10
    probs = model.inference(img, img_mask, max_len)
    print(probs.shape)
     