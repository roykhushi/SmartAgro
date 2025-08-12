# src/train.py
import argparse
import yaml
from data_preprocessing import load_and_preprocess
from model_utils import train_rf, save_model, evaluate
from sklearn.model_selection import train_test_split

def main(cfg):
    X, y, preprocessor, label_encoder = load_and_preprocess(cfg['data']['train_csv'])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_rf(X_train, y_train, cfg.get('model'))
    metrics = evaluate(model, X_val, y_val)

    save_model(model, cfg['paths']['model_out'])
    save_model(preprocessor, cfg['paths']['preproc_out'])
    save_model(label_encoder, cfg['paths']['label_enc_out'])

    print("Validation Metrics:", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='src/config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
