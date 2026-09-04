# Demir Sayım — fotoğraftan inşaat demiri sayımı

İstifin ucunun fotoğrafını çek, kesitleri say, eksik/fazlayı dokunarak düzelt,
irsaliye numarasıyla kaydet. Her şey cihazda: sunucu yok, hesap yok, internet yok.

- Ürün tanımı ve fazlar: [SPEC.md](SPEC.md)
- Çalışma kuralları, komutlar, kazınmış dersler: [CLAUDE.md](CLAUDE.md)
- Tüm eşikler: [config.yaml](config.yaml) (kodda sihirli sayı yok)

## Ne durumda

| Faz | Durum |
|---|---|
| 1 — klasik CV baseline | tamam — sentetik sette MAPE %0.76 (dev) / %0.30 (holdout) |
| 2 — ön-etiketleme araçları | tamam, gerçek fotoğraf bekliyor |
| 3 — YOLO + karolama | tamam — sentetik sette MAPE %0.00, ±2 içinde %100 |
| 4 — Core ML + iOS | tamam — simülatörde uçtan uca çalışıyor, paket 6.9 MB |
| 5 — saha testi | ölçüm altyapısı hazır, gerçek kullanıcı bekliyor |

**Bütün doğruluk rakamları sentetik veriden.** `data/eval/` klasörüne SPEC §7
adlandırmasıyla 50 gerçek fotoğraf girene kadar gerçek kabul kriteri ölçülmemiş
sayılır.

## Kurulum

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Core ML dışa aktarımı ayrı bir ortam ister — coremltools'un yerel eklentileri
Python 3.14 için yayınlanmıyor:

```bash
brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv .venv-coreml
.venv-coreml/bin/python -m pip install "torch<2.8" ultralytics coremltools opencv-python numpy pyyaml
```

## Hızlı deneme

```bash
.venv/bin/python tests/make_synthetic.py --out out/synthetic
.venv/bin/python python/count.py out/synthetic/*degen.png --debug out/dbg --json
.venv/bin/python python/eval.py out/synthetic
```

Diğer komutların tamamı [CLAUDE.md](CLAUDE.md) içinde.
