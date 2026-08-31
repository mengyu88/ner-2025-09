import argparse
import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as config_module
import data_loader
import utils
from model import Model


def evaluate(model, loader):
    model.eval()
    total_ent_r = 0
    total_ent_p = 0
    total_ent_c = 0

    with torch.no_grad():
        for data_batch in loader:
            entity_text = data_batch[-1]
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = [
                data.cuda() for data in data_batch[:-1]
            ]
            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            outputs = torch.argmax(outputs, -1)
            ent_c, ent_p, ent_r, _ = utils.decode(
                outputs.cpu().numpy(), entity_text, sent_length.cpu().numpy()
            )
            total_ent_r += ent_r
            total_ent_p += ent_p
            total_ent_c += ent_c

    return utils.cal_f1(total_ent_c, total_ent_p, total_ent_r)


def predict(model, loader, data, batch_size, output_path):
    model.eval()
    result = []
    i = 0

    with torch.no_grad():
        for data_batch in loader:
            sentence_batch = data[i:i + batch_size]
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = [
                data.cuda() for data in data_batch[:-1]
            ]
            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            outputs = torch.argmax(outputs, -1)
            _, _, _, decode_entities = utils.decode(
                outputs.cpu().numpy(), data_batch[-1], sent_length.cpu().numpy()
            )

            for ent_list, sentence in zip(decode_entities, sentence_batch):
                tokens = sentence["sentence"]
                instance = {"sentence": tokens, "entity": []}
                for ent in ent_list:
                    instance["entity"].append({
                        "text": [tokens[x] for x in ent[0]],
                        "type": config.vocab.id_to_label(ent[1]),
                    })
                result.append(instance)
            i += batch_size

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/food.json")
    parser.add_argument("--checkpoint", default="outputs/food/model.pt")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    config_args = argparse.Namespace(
        config=args.config,
        save_path=None,
        predict_path=None,
        device=args.device,
        dist_emb_size=None,
        type_emb_size=None,
        lstm_hid_size=None,
        conv_hid_size=None,
        bert_hid_size=None,
        ffnn_hid_size=None,
        biaffine_size=None,
        dilation=None,
        emb_dropout=None,
        conv_dropout=None,
        out_dropout=None,
        epochs=None,
        batch_size=None,
        clip_grad_norm=None,
        learning_rate=None,
        weight_decay=None,
        bert_name=None,
        bert_learning_rate=None,
        warm_factor=None,
        use_bert_last_4_layers=None,
        seed=None,
    )

    global config
    config = config_module.Config(config_args)
    config.logger = logging.getLogger("evaluate_checkpoint")
    config.logger.addHandler(logging.NullHandler())

    torch.cuda.set_device(args.device)
    datasets, ori_data = data_loader.load_data_bert(config)
    dev_loader = DataLoader(
        datasets[1],
        batch_size=config.batch_size,
        collate_fn=data_loader.collate_fn,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        datasets[2],
        batch_size=config.batch_size,
        collate_fn=data_loader.collate_fn,
        shuffle=False,
        num_workers=4,
    )

    model = Model(config).cuda()
    model.load_state_dict(torch.load(args.checkpoint))

    dev_f1, dev_p, dev_r = evaluate(model, dev_loader)
    test_f1, test_p, test_r = evaluate(model, test_loader)
    predict(model, test_loader, ori_data[-1], config.batch_size, config.predict_path)

    print(f"DEV entity:  F1={dev_f1:.4f} P={dev_p:.4f} R={dev_r:.4f}")
    print(f"TEST entity: F1={test_f1:.4f} P={test_p:.4f} R={test_r:.4f}")
    print(f"Predictions: {config.predict_path}")


if __name__ == "__main__":
    main()
