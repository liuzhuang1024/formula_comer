from comer.model.comer import CoMER
import yaml
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    params = yaml.full_load(open(args.config, 'r'))

    model = CoMER(
        **params['model']
    )
    print(model)