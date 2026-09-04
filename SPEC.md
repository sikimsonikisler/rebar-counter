# Fotoğraftan İnşaat Demiri Sayımı — Proje Spesifikasyonu

> Bu dosya projenin brief'idir. Konuşma dili Türkçe, kod ve tanımlayıcılar İngilizce.
> Bu dosyadaki tüm komutlar **repo kökünden** çalıştırılır.

---

## 0. Bir bakışta

Şantiyeye demir yüklü kamyon gelir. İrsaliyede "420 adet Ø12" yazar. Kontrol eden kişi istifin ucuna bakıp elle sayar: yavaş, hatalı, ve tartışmalı. Bu uygulama istifin ucunun fotoğrafını çeker, kesitleri sayar, kullanıcı eksik/fazlayı dokunarak düzeltir, sonuç irsaliye numarasıyla kaydedilir.

**Ürün cümlesi:** Demir istifinin fotoğrafını çek, 10 saniyede doğrulanmış sayıyı al.

---

## 1. Neden bu nesne seçildi

Tek nesne türüne odaklanmak zorunlu — "her şeyi sayan uygulama" hedefi bu projeyi öldürür. İnşaat demiri kesiti şu üç nedenden seçildi:

1. **Şekil önsel bilgisi çok güçlü.** Kesitler daire, çapları birbirine çok yakın, aynı düzlemde diziliyor. Bu, etiketli veri toplamadan bile klasik görüntü işlemeyle anlamlı bir doğruluk yakalanabileceği anlamına geliyor — yani birinci gün çalışan bir şeyin olur.
2. **Acı parayla ölçülüyor.** Demir kilo üzerinden satılıyor; 20 çubukluk sayım hatası ciddi para. Sayan kişi de zaten bu işi yapıyor, yani yeni alışkanlık öğretmiyorsun, var olan işi hızlandırıyorsun.
3. **Motor genişleyebilir.** Boru, profil, kereste, tomruk — hepsi "dairesel/tekrarlı kesit sayımı" ailesinde. v1 demirle çalışırsa, aynı pipeline'a ikinci sınıf eklemek yeni proje değil, yeni veri seti.

Reddedilen alternatifler: **koli/palet** (dikdörtgen + üst üste binme = oklüzyon problemi çok ağır), **tomruk** (kesitler düzensiz, arka plan gürültülü, kitle küçük).

---

## 2. Değişmez kısıtlar

Bunlar tartışmaya kapalı. Her teknik karar bunlara uymak zorunda:

| Kısıt | Sebep |
|---|---|
| Sunucu yok | Aylık maliyet ve bakım yükü sıfır kalsın; tek kişilik proje |
| Kullanıcı hesabı yok | Şifre sıfırlama, GDPR/KVKK talebi, destek trafiği doğmasın |
| İnternet gerekmez | Şantiyede kapsama yok; uygulama uçak modunda tam çalışmalı |
| Tüm çıkarım cihazda | Core ML; fotoğraf telefondan çıkmaz — bu aynı zamanda satış argümanı |
| Tek nesne sınıfı (v1) | Kapsam koruması |
| Üçüncü parti API yok | Kırılacak bağımlılık olmasın |

---

## 3. Kullanıcı akışı

1. **Kamera** — rehber çerçeve ve uyarı: "istifin ucu kadraja sığsın, mümkün olduğunca dik çek".
2. **Çekim** — tek fotoğraf. (Canlı video sayımı v1'de yok.)
3. **Sayım** — model çalışır, her tespit üzerine bir nokta konur, üstte büyük puntoyla sayı görünür.
4. **Düzeltme** — kullanıcı pinch-zoom yapar; boş kesite dokunur → nokta ekler, yanlış noktaya dokunur → siler. Geri al butonu var. Sayı canlı güncellenir.
5. **Kayıt** — irsaliye no, çap, not alanları. Kaydet.
6. **Paylaşım** — işaretli görsel + sayı + not tek görüntü olarak dışa aktarılır (WhatsApp'a gider).

> **Bu akışın kalbi 4. adım.** %92 doğruluk bile, düzeltmesi 10 saniye sürüyorsa kullanıcı kazanıyor. Model doğruluğunu kovalarken düzeltme arayüzünü ihmal etme — ürünün gerçek değeri orada.

---

## 4. Teknik yaklaşım ve gerekçesi

### Neden önce klasik CV, sonra ML

Doğrudan YOLO eğitmek daha yüksek doğruluk verir ama işin en yavaş kısmıyla başlamanı zorlar: birkaç yüz fotoğraf çekip elle etiketlemek. Sıralama şöyle kurulacak:

```
Klasik CV baseline  →  baseline'ın çıktısı ÖN-ETİKET olarak kullanılır
                    →  insan sadece düzeltir (etiketleme 5-10x hızlanır)
                    →  bu etiketlerle YOLO eğitilir
                    →  YOLO baseline'ı geçer, baseline yedek/karşılaştırma olarak kalır
```

Yani klasik CV hem birinci gün çalışan bir ürün veriyor, hem de ML fazının veri maliyetini düşüren araç oluyor. Bu yüzden "ikisinden birini seç" değil, **ikisi ardışık**.

### Klasik CV pipeline (Faz 1)

```
gri tonlama
  → CLAHE (yerel kontrast; gölge ve parlama dengelenir)
  → bilateral filter (kenarı koruyan gürültü azaltma)
  → adaptive threshold  VEYA  Canny  (config'ten seçilebilir)
  → morphological opening (küçük gürültüyü sil)
  → distance transform + watershed  ← BİTİŞİK KESİTLERİ AYIRAN ADIM, kritik
  → contour çıkarımı
  → filtre: alan aralığı, dairesellik (4πA/P²), medyan yarıçapa göre outlier atma
  → sayım + merkez noktaları
```

`HoughCircles` alternatif bir dal olarak dursun, ama varsayılan watershed olsun: demirler istifte birbirine değdiği için Hough bitişik kesitleri kaçırıyor.

**Otomatik ölçek tahmini:** ilk geçişte bulunan kesitlerin medyan yarıçapı hesaplanır, ikinci geçişte bu yarıçapın ±%40 dışındaki adaylar elenir. Kullanıcıdan çap bilgisi istemeye gerek kalmaz.

### ML modeli (Faz 3)

- **Tiling zorunlu.** Bir fotoğrafta 300-600 kesit olabiliyor; 640×640'a küçültülünce her kesit birkaç piksele düşer ve model göremez. Görüntü **örtüşmeli karolara** bölünecek (örn. 640×640, %20 örtüşme), her karoda tespit yapılacak, sonuçlar global koordinata çevrilip NMS ile birleştirilecek. Bu yaklaşım literatürde SAHI (Slicing Aided Hyper Inference) olarak geçiyor.
- **Model:** YOLO ailesinden nano/small boyut. Sebep: Core ML dönüşümü sorunsuz, telefonda hızlı, tek sınıf için zaten fazlasıyla kapasiteli.
- **Neden density-map değil:** çok yoğun sahnelerde density regresyon daha iyi sayar ama çıktısı sadece bir sayıdır, nokta listesi vermez. Nokta listesi olmadan düzeltme arayüzü kurulamaz — ve düzeltme arayüzü ürünün kalbi. Bu yüzden **tespit tabanlı** seçildi.
- **Augmentation:** rotasyon (0-360°, kesitler dönmeye duyarsız), ölçek, parlaklık/kontrast, motion blur, JPEG artefaktı, ıslaklık/parlama simülasyonu.

---

## 5. Repo yapısı

```
rebar-counter/
  SPEC.md                  # bu dosya
  CLAUDE.md                # çalışma kuralları (bkz. §10)
  config.yaml              # TÜM eşikler burada, kodda sihirli sayı yok
  data/
    raw/                   # ham fotoğraflar (git'e girmez, .gitignore)
    labels/                # etiketler
    eval/                  # SABİT test seti — asla eğitime veya eşik ayarına girmez
  python/
    count.py               # CLI giriş noktası
    pipeline/
      preprocess.py        # gri, CLAHE, filtre
      segment.py           # threshold/canny, morfoloji, watershed
      detect.py            # contour çıkarımı ve filtreleme
      draw.py              # debug ve sonuç görselleri
    ml/
      tile.py              # örtüşmeli karolama + NMS birleştirme
      train.py
      predict.py
      export_coreml.py
    eval.py                # metrik hesaplama
    prelabel.py            # baseline çıktısını etiket formatına çevirir
  ios/
    RebarCounter/          # SwiftUI uygulaması (Faz 4)
  tests/
```

---

## 6. Fazlar

Her faz bir öncekinin kabul kriterini geçmeden başlamaz.

### Faz 0 — Veri toplama protokolü (arka planda sürekli)

Bu faz kod değil, disiplin. Detay §7'de.

**Kabul:** en az 50 fotoğraf, her birinin elle sayılmış gerçek değeri kayıtlı. Bu 50 fotoğraf `data/eval/` klasörüne konur ve Faz 1'in test seti olur — sonradan gelen fotoğraflar eğitim setine gider, bu 50'si asla değişmez.

### Faz 1 — Python CLI baseline

Görevler:

- `count.py` CLI'ı yaz. Sözleşme:
  ```
  python count.py <image_path> [--config config.yaml] [--debug <out_dir>] [--json]
  ```
  Çıktı (stdout, `--json` ile):
  ```json
  { "count": 412, "points": [[x, y, r]], "method": "classical", "elapsed_ms": 340 }
  ```
  `--debug` verilirse her ara adımın görselini `out_dir`'e yazsın (`01_gray.png`, `02_clahe.png`, ... `09_result.png`). Bu görseller ayar yapmanın tek yolu, opsiyonel değil.
- `eval.py` yaz:
  ```
  python eval.py data/eval/ --config config.yaml
  ```
  Çıktı tablosu: dosya adı, gerçek, tahmin, hata, hata%. Altta özet: **MAE**, **MAPE**, **±2 adet içinde kalma oranı**, ortalama süre.
- Tüm eşikleri `config.yaml`'a taşı. Kodda hiçbir sabit sayı kalmasın.

**Kabul kriteri:** 50 fotoğraflık eval setinde **MAPE ≤ %10**.

### Faz 2 — Ön-etiketleme ve etiket seti

- `prelabel.py`: baseline çıktısını etiketleme aracının okuyabileceği formata çevir.
- Etiketleme aracı olarak label-studio veya CVAT kur; ön-etiketleri içeri aktar, insan sadece düzeltsin.
- Hedef: **300+ fotoğraf, tamamı düzeltilmiş etiketli.** `data/eval/` bunun dışında ve dokunulmaz kalır.

**Kabul:** eğitim seti hazır, eval seti eğitim setinden tamamen ayrık (dosya adı bazında doğrulanmış).

### Faz 3 — ML modeli

- `tile.py`: örtüşmeli karolama, karo bazında tahmin, global koordinata dönüşüm, NMS ile birleştirme. Tek başına test edilebilir olsun.
- `train.py`: YOLO nano/small eğitimi, augmentation ayarları `config.yaml`'dan.
- `predict.py`: `count.py` ile aynı JSON sözleşmesini döndürsün, tek farkı `"method": "ml"`.
- `count.py`'a `--method classical|ml` bayrağı ekle. **Aynı CLI, değiştirilebilir motor.**
- `eval.py` ile iki motoru aynı eval setinde karşılaştıran tablo üret.

**Kabul kriteri:** **MAPE ≤ %3** ve fotoğrafların **≥%90'ında hata ≤ ±2 adet**.

### Faz 4 — Core ML dönüşümü ve iOS uygulaması

- `export_coreml.py`: `coremltools` ile dönüşüm, float16 nicemleme. Dönüşüm sonrası Python ve Core ML çıktılarını aynı 20 görüntüde karşılaştıran doğrulama scripti — sayı farkı 0 olmalı.
- SwiftUI uygulaması: `CameraView`, `CountingService`, `ResultView`, `RecordListView` / `RecordDetailView`, paylaşım.
- Ayarlar ekranı: yok. v1'de ayar yok.

**Kabul kriteri:** iPhone'da çıkarım **< 1.5 sn**, uygulama boyutu **< 100 MB**, uçak modunda tam çalışıyor.

### Faz 5 — Saha testi

- 3 gerçek kullanıcı, en az 20 gerçek sayım.
- Kaydedilecek: uygulamayla toplam süre, elle sayım süresi, iki sayının farkı.

**Kabul kriteri:** ortalama toplam süre **< 30 sn** ve uygulama+düzeltme sonucu elle sayımdan daha az hatalı.

---

## 7. Veri toplama protokolü

Modelin tavanını veri belirliyor, mimari değil. Çeşitlilik matrisi:

| Boyut | Kapsanacak değerler |
|---|---|
| Çap | 8, 10, 12, 14, 16, 20 mm |
| Işık | gündüz açık, gölge, bulutlu, flaşlı, karşı ışık |
| Açı | 0°, ~10°, ~20° eğim |
| Yüzey | temiz, paslı, ıslak, boyalı uç, etiketli/bantlı |
| İstif | seyrek, sıkı, düzensiz, birden fazla demet bir arada |
| Arka plan | toprak, beton, kamyon kasası, başka demir yığını |

Dosya adlandırma (eval scriptleri bunu parse edecek):

```
YYYYMMDD_hhmm_<gerçeksayı>_<çap>_<koşul>.jpg
örn: 20260910_1430_412_12_islak.jpg
```

Gerçek sayı **iki kez elle** sayılıp doğrulanacak; iki sayım tutmuyorsa üçüncü kez sayılıp fotoğraf ancak öyle sete girecek. Yanlış ground truth, modelin öğrenebileceği en pahalı hatadır.

---

## 8. Değerlendirme

- **MAE** — ortalama mutlak hata (adet).
- **MAPE** — ortalama mutlak yüzde hata. **Ana metrik budur.**
- **±2 içinde kalma oranı** — kullanıcı deneyimine en yakın metrik.

Kural: **her değişiklikten sonra `eval.py` çalıştırılır ve MAPE raporlanır.** Bir değişiklik MAPE'yi kötüleştiriyorsa geri alınır. Eşik ayarı yalnızca eğitim/geliştirme setinde yapılır, `data/eval/` asla ayar için kullanılmaz.

---

## 9. Kapsam dışı (v1'de kesinlikle yok)

- Bulut, hesap, cihazlar arası senkron
- Birden fazla nesne türü
- Video akışında canlı sayım
- Ağırlık/tonaj hesabı
- Çoklu dil (v1 sadece Türkçe)
- Ayarlar ekranı, tema, onboarding turu

---

## 10. `CLAUDE.md`

Repo kökündeki `CLAUDE.md` dosyasına bakınız.

---

## 12. v2 fikirleri (şimdi yapma, sadece kayıt)

- **Ağırlık hesabı:** çelik çubuk için metre başına ağırlık ≈ çap²/162 kg. Adet × boy × bu katsayı ile irsaliyedeki tonajı doğrulama.
- İkinci nesne sınıfı: boru / profil.
- Zaman içinde biriken düzeltmelerden yeniden eğitim (cihazda toplanır, kullanıcı isterse dışa aktarır — otomatik gönderim yok).
- Ekip modu: aynı irsaliyeyi iki kişi sayıp karşılaştırma (QR ile cihazdan cihaza aktarım).
