# CALIBRI — сессия №10 (2026‑11‑01)
## Научная карта: квантование, MoE, кодеки, I/O, мобильное исполнение

**Ключевая установка:** 80 ГБ — физический предел хранилища телефона,
а НЕ лимит оперативной памяти и НЕ доказательство, что модель будет быстро работать.
В телефон устанавливать только опубликованный, проверенный и совместимый
model-bundle. Ни одна из экспериментальных версий CALIBRI сегодня этому не соответствует.

### A. Научная карта — работы для воспроизводимого сравнения

Приоритет: P0 — необходимо воспроизвести; P1 — непосредственно полезно;
P2 — смежная идея/требует проверки переносимости. Ссылки ведут к первичным работам.

| № | Работа (год) | Метод/наблюдение | Что проверить для CALIBRI | Priority |
|---|---|---|---|---|
| 1 | [QMoE (MLSys 2024)](https://proceedings.mlsys.org/paper_files/paper/2024/hash/c74b624843218d9b6713fcf299d6d5e4-Abstract-Conference.html) | Тернарное сжатие MoE и fused GPU decode/matvec <1 bpw | Тот же quantized checkpoint и железо, не синтетику | P0 |
| 2 | [Switch Transformers (JMLR 2022)](https://www.jmlr.org/beta/papers/v23/21-0998.html) | Top-1 routing MoE, capacity | Маршрутизация, распределение нагрузки экспертов, фактические active weight reads | P0 |
| 3 | [GPTQ (2023)](https://arxiv.org/abs/2210.17323) | Аппрокс. second-order квантование | Перестать задавать 88.5% нулей вручную; Hessian-aware групповой PTQ | P0 |
| 4 | [MoEQuant (ICML 2025)](https://proceedings.mlr.press/v267/chen25aa.html) | Балансировка калибровки и affinity между экспертами | Редкие эксперты должны получать достаточно calibration samples | P0 |
| 5 | [Efficient Quantization of MoE (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7b97d09f0ab0f52eab856b0ed1122456-Abstract-Conference.html) | Expert-sensitive mixed precision с теоретическим обоснованием | Не понижать точность редких, но критичных экспертов | P0 |
| 6 | [PuzzleMoE (ICML 2026)](https://proceedings.mlr.press/v306/zhao26ay.html) | Fine-grained expert merging + bit-packed inference | Отделить shared и specialized параметры без лишних масок | P0 |
| 7 | [ZipMoE (ICML 2026)](https://proceedings.mlr.press/v306/yang26ds.html) | Lossless сжатие плюс cache-affinity scheduling для on-device MoE | Тестировать CLB on-device, workload-affine кеш вместо слепого LRU | P0 |
| 8 | [LLM in a Flash (ACL 2024)](https://machinelearning.apple.com/research/efficient-large-language) | Windowing, row-column bundling, flash-friendly contiguous reads | Измерять p95 чтения страниц весов из Android storage | P0 |
| 9 | [AWQ (MLSys 2024)](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) | Масштабирование активационно-салиентных каналов | Надёжный mobile INT4 baseline | P0 |
| 10 | [SpQR (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1787533e171dcc8549cc2eb5a4840eec-Abstract-Conference.html) | Sparse high-precision outlier sidecar | Проверять sparse residual вместо 100% ternary и учитывать метаданные | P0 |
| 11 | [DecDEC (OSDI 2025)](https://www.usenix.org/conference/osdi25/presentation/park-yeonhong) | Динамическое добавление salient residual для low-bit слоя | Разделить компактный base и защищённый error tail, считать bandwidth | P0 |
| 12 | [T-MAC (2024)](https://arxiv.org/abs/2407.00088) | LUT CPU matmul без развёртывания квантованных весов | Сравнить с packed INT4 на ARM, реально учитывать lookup cost | P0 |
| 13 | [AQLM (ICML 2024)](https://proceedings.mlr.press/v235/egiazarian24a.html) | Multi-codebook additive quantization с адаптацией под активации | Векторная rate–distortion модель, decode kernel cost | P1 |
| 14 | [QuIP# (ICML 2024)](https://proceedings.mlr.press/v235/tseng24a.html) | Randomized Hadamard + E8 lattice codebooks | Предобработка каналов до квантования; не считать новое открытие | P1 |
| 15 | [QuaRot (NeurIPS 2024)](https://papers.nips.cc/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html) | Сохраняющие функцию повороты убирают outliers | Тест “rotation vs cost of transform + quantization” | P1 |
| 16 | [OmniQuant (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c6483c8a68083af3383f91ee0dc6db95-Abstract-Conference.html) | Learnable clipping/equivalent transformations | Совместный подбор clipping/scale/bitwidth до формирования CLB | P1 |
| 17 | [SqueezeLLM (ICML 2024)](https://proceedings.mlr.press/v235/kim24f.html) | Dense + sparse + sensitivity | Dense-small + sparse corrections по качеству/байтам | P1 |
| 18 | [SparseGPT (ICML 2023)](https://proceedings.mlr.press/v202/frantar23a) | One-shot second-order pruning | Структурная разреженность вместо принудительных 88.5% нулей | P1 |
| 19 | [MoBE (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/431d53d513461ff155d5bc8faa9a440c-Abstract-Conference.html) | Basis matrices общие для экспертов | Проверить общий базис/коэффициенты с учётом фактического доступа | P1 |
| 20 | [D²-MoE (ICML 2025)](https://proceedings.mlr.press/v267/gu25c.html) | Shared core + SVD delta | Измерить реальную conditional entropy до объединения экспертов | P1 |
| 21 | [CAMERA (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39957) | Micro-expert cross-matrix compression | Перенести оптимизацию с матрицы на совместные up/down микроблоки | P1 |
| 22 | [Model Folding (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f102c495b31bcc7b2a11c27de75475fc-Abstract-Conference.html) | Data-free folding похожих нейронов | Контролируемое approx-fusion и worst-case/OOD | P1 |
| 23 | [Sub-MoE (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39464) | Subspace expert clustering and merging | Проверить функциональное, а не просто весовое сходство | P1 |
| 24 | [Neural Weight Compression (2025)](https://arxiv.org/abs/2510.11234) | Neural codec весов с loss, ориентированным на модель | Исследовать learned entropy coding *без обязательного decode-all* | P1 |
| 25 | [LLMCodec (2026)](https://arxiv.org/abs/2606.05861) | VVC/H.266 для квантованных весов | Замерить archive size vs CPU decoding overhead, не подменять скорость сжатием | P1 |
| 26 | [CoTAM (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0cf3e7eefb9d643e93e16ff1d94090a7-Abstract-Conference.html) | Feature-level perceptual importance в image compression | Перенос rate-distortion с пикселей на downstream логиты/роутеры | P1 |
| 27 | [FlashMoE (2026)](https://arxiv.org/abs/2601.17063) | SSD offloading + ML expert cache | Реальная адаптивная политика чтения из мобильной flash | P1 |
| 28 | [CasMoE (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/39816) | Offline/online expert prefetch | Предзагрузка больших блоков до фактического выбора эксперта | P1 |
| 29 | [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | CPU/SSD offload, expert cache | Контрольный алгоритм offloading/prefetch и горячих экспертов | P1 |
| 30 | [QServe (MLSys 2025)](https://proceedings.mlsys.org/paper_files/paper/2025/hash/fbe2b2f74a2ece8070d8fb073717bda6-Abstract-Conference.html) | Weight+activation+KV formats, GPU kernel design | Decoder operations/byte, KV cache cost, whole token budget | P1 |
| 31 | [BitNet (JMLR 2025)](https://www.jmlr.org/beta/papers/v26/24-2050.html) | Native ternary models обучают с нуля | Нельзя переносить на PTQ тернаризацию; искать готовые native low-bit weights | P1 |
| 32 | [BitNet b1.58 2B4T (2025)](https://arxiv.org/abs/2504.12285) | Реально обученная low-bit модель | Обязательный тест для альтернативы тяжёлому Switch на телефоне | P1 |
| 33 | [FlashAttention-2 (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html) | I/O-aware tiling | Холодные/горячие блоки, не повышать I/O на каждом token | P2 |
| 34 | [EAGLE (ICML 2024)](https://proceedings.mlr.press/v235/li24bt.html) | Speculative decoding | Снижение latency отдельно от сжатия; оптимизация общей эффективности | P2 |
| 35 | [MENDS-MoE (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.2041/) | Next-router order preservation | Проверять маршрутизацию следующего слоя при низких битах | P1 |

### B. Что здесь уже известно и не должно выдаваться за новизну

- Словарь + тернарная матрица + fused matvec — QMoE.
- Узкоаппаратный LUT matvec — T-MAC.
- Общий эксперт и SVD residual — D²-MoE / MoBE.
- Переупорядочивание/поворот каналов — QuIP#, QuaRot, Folding.
- Sparse residual/high precision outliers — SpQR, SqueezeLLM, DecDEC.
- Использовать video codec на весах — LLMCodec.
- On-device lossless сжатие + cache scheduling — ZipMoE.
- Программно ограничить передаваемые данные по буферу/80 ГБ — общая инженерия.

### C. Проверяемые гипотезы CALIBRI после обзора

**Гипотеза CALIBRI-SIGNAL (из image coding, neural weight compression):**
не переводить матрицу буквально в PNG; использовать преобразование канального
пространства и многомасштабную организацию, если выполняются одновременно:
  1. причинно согласованная форма вычисления с перемещёнными весами;
  2. размер с реальными метаданными меньше CL8, AQLM/QuIP# baseline;
  3. цена преобразования не больше выигранного времени чтения;
  4. изменения качества проверены на независимой валидации и OOD.

**Гипотеза CALIBRI-RD (из information theory/CoTAM):**
использовать routing-aware Lagrangian rate-distortion:
```text
min   Σ_e p_e E[|| (W_e-W_hat_e) x_e ||²]
      + λ · total_real_bytes
      + μ · p95_compute_latency
      + ν · actual_storage_bytes_read_per_token
s.t.  worst_rare_expert_error ≤ cap
      next_router_topk_agreement ≥ threshold
      heldout task quality ≥ gate
```
Это постановка эксперимента, а не доказанное новое решение.
Критическая новизна должна быть в реализации оператора/биtstream,
дающей реальное Pareto-improvement против научных baseline.

**Гипотеза CALIBRI-BITPLANE (из видео и wavelet, lossless):**
проверить разделение 3-значной поддержки на sign/occupancy bitplanes;
попробовать межэкспертный delta только если conditional entropy, ratio
и реальные замеры ридера показывают экономию. Отказаться от delta при
low-MI (на реальном Pythia EXACT-LIFT увеличил размер на 29–51%).

**Гипотеза CALIBRI-BLOCK/FLASH:**
формат должен быть непосредственно addressable без decompress-all.
При реальной routing трассе измерять `bytes_read/token`, p50/p95,
page faults, storage throughput и cache hit. Никакие результаты
desktop GPU не обосновывают производительность Android автоматически.

### D. Протокол исследований

1. FP16/BF16 baseline качества реальной MoE, pinned revision.
2. Официальный QMoE checkpoint и сравнение побитового восстановления.
   При недоступности checkpoint не подменять baseline синтетикой.
3. Calib train/val/test: expert-balanced (редкий expert тоже проверяется).
4. Минимум: CL8; int4, int8; QMoE; AQLM/QuIP# как возможно; decode/exec latency.
5. Аblation: +outlier residual, +routing-weighted allocation, +bitplanes,
   +prefetch, +block scheduling, каждая по отдельности.
6. Quality: seq2seq loss, task accuracy, router disagreement, OOD, not
   merely operator relative RMSE.
7. Android: real filesystem free, 80GB storage budget, minimum OS reserve,
   no loading into phone until verified signed bundle and compatible reader.
8. Failure cases and negative experiments must be reported, not hidden.

### E. Отдельная кнопка “Обновить память”

- `Обновить Jessica`: только прежний APK updater, 250MiB limit remains.
- `Обновить память`: independent state and action; check actual free disk
  and current models; if no trusted model release, show **NO APPROVED MODEL**
  and never download speculative weights.
- ModelReleaseGate currently fail-closed with empty trusted-manifest allowlist.
- No automatic merge to main/Render, no live user data/weights, no APK publish.

### F. Критерии прогресса сессии №11

Do not claim "0.8 bits, quality-preserved" until exact official QMoE
quantized tensors, entire model metric, and runtime comparability are verified.
Try to show a **single real Pareto-improvement** in (quality,size,latency)
rather than an ever-larger catalogue of plausible but unverified compression ideas.

### G. Дополнительные работы вне узкой темы LLM — кодеки и физические носители

| № | Работа | Заимствуемая идея | Граница переносимости |
|---|---|---|---|
| 36 | [Variational image compression with a scale hyperprior (ICLR 2018)](https://research.google/pubs/variational-image-compression-with-a-scale-hyperprior/) | Обучаемая условная модель энтропии с компактным hyperprior | Стоимость hyperprior + случайного доступа должна окупаться |
| 37 | [DVC: End-to-End Deep Video Compression (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Lu_DVC_An_End-To-End_Deep_Video_Compression_Framework_CVPR_2019_paper.html) | Предиктор + остаток + оптимизация rate-distortion | Веса разных экспертов — не последовательные видеокадры |
| 38 | [EnCodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) | Residual vector quantization + streaming decoder | Декодирование весов для random-access отличается от аудиопотока |
| 39 | [ContextGS: Compact 3D Gaussian Splatting (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5c20ca4b0b20b0bd2f1d839dc605e70f-Abstract.html) | Многоуровневое условное кодирование структурированных 3D-якорей | Сопоставление экспертов/нейронов — гипотеза, не данность |
| 40 | [CAT-3DGS: Rate-Distortion-Optimized 3D Compression (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/41e638f36d9cb052669d30f559e7c0b7-Abstract-Conference.html) | Spatial/feature hyperprior + rate-distortion allocation | Размер side-channel и задержка чтения |
| 41 | [Photonic Mixture-of-Experts (Nature Communications 2026)](https://www.nature.com/articles/s41467-026-73983-4) | Физические параллельные expert cores + routing | Требует нового оптического железа; не кодек для памяти Android |
| 42 | [Universal photonic AI acceleration (Nature 2025)](https://www.nature.com/articles/s41586-025-08854-x) | Optical tensor processing | Не снижает информацию, необходимую для хранения цифровых весов |
| 43 | [LLM in a Flash (ACL 2024)](https://machinelearning.apple.com/research/efficient-large-language) | Память flash как отдельный уровень и крупные последовательные чтения | На телефоне необходимо проверить I/O, износ и тепловые ограничения |

**Вывод по аналогиям:** перспективны conditional entropy modeling, rate–distortion,
vector residual coding и крупноблочный I/O. Оптическая обработка — отдельная
аппаратная область, не способ магического сжатия 80 ГБ весов в обычной флеш-памяти.
