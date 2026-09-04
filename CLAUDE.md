# Çalışma kuralları

- Benimle Türkçe konuş. Kod, değişken adları, commit mesajları ve yorumlar İngilizce.
- Her pipeline adımı ayrı ve tek işi olan bir fonksiyon olsun. Adımlar `--debug` ile
  ara görsellerini diske yazabilmeli.
- Kodda sihirli sayı yasak. Her eşik, çekirdek boyutu, oran `config.yaml`'da durur.
  Telefonun ihtiyaç duyduğu değerler `export_coreml.py` tarafından aynı dosyadan
  `counting_config.json`'a yazılır — Swift tarafında elle sayı tanımlama.
- Görüntü işleme değişikliğinden sonra MUTLAKA `eval.py` çalıştır ve MAPE'yi önceki
  değerle karşılaştırarak raporla. Kötüleşme varsa değişikliği geri al.
- `data/eval/` klasörü kutsaldır: eğitimde, eşik ayarında veya hata ayıklamada kullanılmaz.
  `tune.py` bu klasörde çalışmayı reddeder, `ml/dataset.py` sızıntıyı hata olarak keser.
- Yeni bağımlılık eklemeden önce sor. Liste kısa: opencv-python, numpy, ultralytics,
  coremltools, pyyaml, pillow.
- **İki sanal ortam var.** `.venv` (Python 3.14) her şeyi yapar; `.venv-coreml`
  (Python 3.12) yalnızca Core ML dışa aktarımı içindir. coremltools'un yerel
  eklentileri (`libcoremlpython`, `libmilstoragepython`) Python 3.14 için yok —
  o yorumlayıcıda ne model yazılabiliyor ne çalıştırılabiliyor.
- Sunucu, veritabanı sunucusu, API anahtarı gerektiren hiçbir çözüm önerme — proje
  tanımı gereği tamamen çevrimdışı. (Model ağırlıklarının bir kereliğine indirilmesi
  geliştirme zamanına aittir; uygulamada ağ kodu yok.)
- Bir fazın kabul kriteri geçilmeden sonraki faza geçme. Geçtiğini `eval.py` çıktısıyla kanıtla.
- Uzun açıklama yapma; ne yaptığını kısa söyle, kodu göster.

## Komutlar

Tüm komutlar repo kökünden. Sanal ortam `.venv/`.

```bash
# Faz 1 - klasik CV
.venv/bin/python python/count.py <image> --debug out/dbg --json
.venv/bin/python python/eval.py out/synthetic --methods classical,hough
.venv/bin/python python/tune.py out/synthetic --rounds 2

# Sentetik veri (gerçek fotoğraf gelene kadar tek doğrulama kaynağı)
.venv/bin/python tests/make_synthetic.py --out out/synthetic                    # 8 sabit varyant
.venv/bin/python tests/make_synthetic.py --out out/synth_train/images \
    --labels out/synth_train/labels --random 80 --seed 500                      # etiketli eğitim seti

# Faz 2 - ön-etiketleme
.venv/bin/python python/prelabel.py data/raw --out data/labels --format yolo
.venv/bin/python python/prelabel.py data/raw --out data/labels --format label-studio
.venv/bin/python python/ml/dataset.py --check-only                              # eval/eğitim ayrıklığı
.venv/bin/python python/ml/dataset.py --out data/yolo                           # karo tabanlı veri seti

# Faz 3 - model
.venv/bin/python python/ml/train.py --data out/synth_yolo/dataset.yaml --device mps
.venv/bin/python python/eval.py out/synthetic --methods classical,ml --weights runs/rebar/weights/best.pt

# Faz 4 - Core ML + iOS   (DİKKAT: dışa aktarım .venv-coreml ile çalışır)
.venv-coreml/bin/python python/ml/export_coreml.py --weights runs/rebar/weights/best.pt \
    --verify-folder out/synth_verify
xcodebuild -project ios/RebarCounter/RebarCounter.xcodeproj -scheme RebarCounter \
    -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
    -derivedDataPath ios/RebarCounter/build CODE_SIGNING_ALLOWED=NO build

# Testler
.venv/bin/python -m pytest tests/ -q
xcodebuild -project ios/RebarCounter/RebarCounter.xcodeproj -scheme RebarCounter \
    -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
    -derivedDataPath ios/RebarCounter/build CODE_SIGNING_ALLOWED=NO test
```

## Referans değerler

Bir değişiklikten sonra bunlarla karşılaştır. Hepsi **sentetik** veriden; `data/eval/`
boş olduğu sürece gerçek kabul kriteri ölçülmemiş sayılır.

| set | motor | MAPE | MAE | ±2 içinde | süre |
|---|---|---|---|---|---|
| `out/synthetic` (ayarın yapıldığı set) | classical | %0.76 | 1.50 | %75 | 57 ms |
| `out/synthetic_holdout` (`--seed 100`) | classical | %0.30 | 0.75 | %100 | 57 ms |
| `out/synthetic` | ml | **%0.00** | 0.00 | %100 | 98 ms |
| `out/synthetic_holdout` | ml | **%0.00** | 0.00 | %100 | 98 ms |

Model: YOLO11n, 80 sentetik fotoğraftan üretilen 425 karo üzerinde 38 epoch
(patience ile erken durdu). Doğrulama: P 0.992, R 0.997, mAP50 0.995, mAP50-95 0.909.

Core ML dönüşümü 20 görüntüde PyTorch ile **birebir aynı sayıyı** veriyor (fark 0).
Karo başına 2.1 ms → 48 karoluk (4032×3024) bir fotoğraf ~0.10 sn (Mac ölçümü).
Uygulama paketi 6.9 MB.

## Kazınmış dersler

Bunlar acıyla öğrenildi, geri alma:

1. **Eşikleme penceresi çubuk çapından küçük olmalı.** Pencere bir çubuktan genişse
   her çubuk komşularıyla kıyaslanır ve ortalamadan koyu olan (paslı) çubukların
   tamamı kaybolur. `segment.flatness` bu yüzden `|I − yerel ortalama|` üzerinden
   çalışıyor: ayırıcı ister koyu gölge ister açık arka plan olsun eleniyor, çubuğun
   kendi parlaklığı hiç işe karışmıyor.
2. **Bitişikleri ayırırken gri görüntü kullanma.** Gri görüntü üzerinde watershed,
   parlaklık farkı yüzünden bir çubuğun komşusuna taşmasına izin veriyor
   (dev sette MAPE %0.76 → %23.65). `nearest_seed` saf geometriyle kesiyor.
3. **Karolama olmadan ML çalışmaz.** 400 kesitli bir fotoğraf 640×640'a küçültülünce
   her kesit birkaç piksele iner. `tile.py` ve Swift'teki `Tiler` aynı planı üretmek
   zorunda — `TilerTests.planMatchesPython` bu pariteyi kilitliyor.
4. **Karo başına tespit limiti.** YOLO'nun varsayılan `max_det=300`'ü yoğun bir karoda
   sayımı sessizce kırpar. `ml.nms.max_detections_per_tile` bu yüzden 1200.
5. **Model içi NMS kullanma.** Core ML dışa aktarımı `nms=False` ile yapılır; birleştirme
   Swift'te, `tile.py` ile aynı kurallarla. Model içindeki NMS katmanının kutu limiti
   aynı sessiz kırpma sorununu geri getirir.
6. **`half=True` değil `quantize=16`.** Ultralytics'in eski `half` bayrağı Detect
   başlığının dönüşümünü tamamen kırıyor (`TypeError: only 0-dimensional arrays...`).
   Aynı fp16 sonucu `quantize=16` ile sorunsuz çıkıyor.
7. **NMS'i ızgarayla kovala.** Tüm fotoğrafın birleştirmesinde ~8000 kutu oluyor;
   her çifti karşılaştırmak telefonda saniyeler alır. `Tiler.nms` kutuları çubuk
   boyutunda hücrelere atıp yalnızca komşuları karşılaştırıyor: 8192 kutu 0.035 sn.
8. **Yakınlaştırmada dokunma koordinatı.** Dokunma jesti `scaleEffect`'ten ÖNCEKİ
   görünüme bağlanır; böylece konum dönüşmemiş çerçevede gelir ve tek bir bölmeyle
   piksele çevrilir. Ters matris hesabına gerek yok.

## Faz durumu

- **Faz 1** — tamam. Klasik CV baseline, `--debug` 9 ara görsel, `eval.py`, `tune.py`.
- **Faz 2** — araçlar tamam (`prelabel.py`, `ml/dataset.py`), veri bekliyor.
- **Faz 3** — tamam (sentetik veriyle). `tile.py`, `train.py`, `predict.py`, `--method ml`.
  Kabul kriteri (MAPE ≤ %3, ±2 içinde ≥ %90) geçildi: %0.00 / %100.
- **Faz 4** — tamam. `export_coreml.py` + SwiftUI uygulaması. Simülatörde 420 kesitlik
  sentetik fotoğrafı 420 sayıyor; dokunarak ekleme/silme, geri al, pinch-zoom,
  kayıt, paylaşım görseli çalışıyor. Paket 6.9 MB (<100 MB ✓), ağ kodu yok (✓).
  **Gerçek cihazda çıkarım süresi ölçülmedi** — simülatör Neural Engine kullanmıyor.
- **Faz 5** — ölçüm altyapısı hazır (uygulama süreyi ve elle sayım farkını kaydediyor,
  "Saha Testi" ekranı özetliyor, CSV dışa aktarımı var). Gerçek kullanıcı bekliyor.

**Eksik olan tek şey gerçek fotoğraf.** 50 fotoğraf `data/eval/` klasörüne SPEC §7
adlandırmasıyla girince Faz 1'in gerçek kabul kriteri ölçülebilir; 300+ etiketli
fotoğrafla model gerçek veride yeniden eğitilir.
