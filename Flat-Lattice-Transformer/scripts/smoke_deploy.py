import collections
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from load_data import load_toy_ner
from V0.add_lattice import equip_chinese_ner_with_lexicon
from V0.models import Lattice_Transformer_SeqLabel


def as_batch(instance, field, dtype=torch.long, device=None):
    return torch.tensor([instance[field]], dtype=dtype, device=device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = ROOT / "data" / "smoke"
    bigram_path = ROOT / "embeddings" / "smoke" / "bigram.vec"
    lattice_path = ROOT / "embeddings" / "smoke" / "lattice_mix.vec"
    lexicon = ["食品", "食品安全", "水产品", "安全标准", "食品标准", "标准"]

    datasets, vocabs, embeddings = load_toy_ner(
        str(data_dir),
        char_embedding_path=None,
        bigram_embedding_path=str(bigram_path),
        index_token=False,
        _refresh=True,
        _cache_fp=str(ROOT / "cache" / "smoke_toy_ner"),
    )

    datasets, vocabs, embeddings = equip_chinese_ner_with_lexicon(
        datasets,
        vocabs,
        embeddings,
        lexicon,
        word_embedding_path=None,
        word_char_mix_embedding_path=str(lattice_path),
        lattice_min_freq=1,
        only_train_min_freq=False,
        _refresh=True,
        _cache_fp=str(ROOT / "cache" / "smoke_lattice"),
    )

    sample = datasets["train"][0]
    max_seq_len = max(max(ds["seq_len"]) for ds in datasets.values())
    model = Lattice_Transformer_SeqLabel(
        embeddings["lattice"],
        embeddings["bigram"],
        hidden_size=16,
        label_size=len(vocabs["label"]),
        num_heads=2,
        num_layers=1,
        use_abs_pos=False,
        use_rel_pos=True,
        learnable_position=False,
        add_position=False,
        layer_preprocess_sequence="",
        layer_postprocess_sequence="an",
        ff_size=32,
        scaled=False,
        dropout=collections.defaultdict(float),
        use_bigram=True,
        mode=collections.defaultdict(bool),
        dvc=device,
        vocabs=vocabs,
        rel_pos_shared=True,
        max_seq_len=max_seq_len,
        k_proj=False,
        q_proj=True,
        v_proj=True,
        r_proj=True,
        self_supervised=False,
        attn_ff=False,
        pos_norm=False,
        ff_activate="relu",
        rel_pos_init=1,
        abs_pos_fusion_func="nonlinear_add",
        embed_dropout_pos="0",
        four_pos_shared=True,
        four_pos_fusion="ff_two",
        four_pos_fusion_shared=True,
        use_pytorch_dropout=0,
    )
    model.to(device)

    model.train()
    output = model(
        lattice=as_batch(sample, "lattice", device=device),
        bigrams=as_batch(sample, "bigrams", device=device),
        seq_len=as_batch(sample, "seq_len", device=device).view(-1),
        lex_num=as_batch(sample, "lex_num", device=device).view(-1),
        pos_s=as_batch(sample, "pos_s", device=device),
        pos_e=as_batch(sample, "pos_e", device=device),
        target=as_batch(sample, "target", device=device),
    )
    loss = output["loss"]
    assert torch.isfinite(loss), loss
    print("smoke ok")
    print(f"device={device}")
    print(f"labels={len(vocabs['label'])} lattice_vocab={len(vocabs['lattice'])} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
