# PFI-VTON: Patch Forcing Inpainting cho thử đồ ảo (VITON-HD 1024 × 768)

Đây là README chính của phần VTON. Mọi lệnh chạy từ **thư mục gốc của repo**.
Mô hình cuối là **V50** (`configs/vton_v50_attn_flow.yaml`), gồm:
- DiT PFT-XL/2 hai luồng (person / garment) với attention bất đối xứng;
- garment K/V mịn (8 px) ở các block cuối;
- các loss: CoRAL, decoded RGB/high-pass, uncertainty và attention-flow.

Ablation ở mục 4, đánh giá ở mục 5. Các file đánh giá cụ thể ở [`docs/experiment/`](../docs/experiment/README.md).

Mã nguồn chính:

| File | Nội dung |
| --- | --- |
| [`pfi_model.py`](pfi_model.py) | Mô hình hai luồng, fine garment K/V, attention flow, chế độ `attention_mode` |
| [`pfi_train.py`](pfi_train.py) | Huấn luyện: bộ lấy mẫu timestep (curriculum, `time_mode`), các loss, eval preview |
| [`pfi_sample.py`](pfi_sample.py) | Sampler `euler` / `dual_loop` / `look_ahead`, CFG theo trang phục, composite |
| [`coral.py`](coral.py) | Teacher DINOv3 và CoRAL loss |
| [`data.py`](data.py), [`pairs.py`](pairs.py) | Dataset VITON-HD, danh sách cặp và fingerprint |

## 1. Môi trường và checkpoint tiền huấn luyện

Dùng Python 3.12 và PyTorch phù hợp GPU. Trên máy local có sẵn conda env `vt-env` để chạy test CPU.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-vton.txt
mkdir -p checkpoints
```

| Khóa config | Đường dẫn | Nguồn |
| --- | --- | --- |
| `weights.pft` | `checkpoints/pft-xl_step400k_ema.ckpt` | [PFT-XL/2 công bố](https://ommer-lab.com/files/pft/pft-xl_step400k_ema.ckpt) |
| `weights.vae` | `checkpoints/sd_ae.ckpt` | [Autoencoder CompVis](https://github.com/CompVis/fm-boosting#-usage) |
| `weights.dino` | `checkpoints/dinov3-vits16-pretrain-lvd1689m/` | [DINOv3 ViT-S/16 của Meta](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) |

```bash
curl -fL --retry 3 -o checkpoints/pft-xl_step400k_ema.ckpt \
  'https://ommer-lab.com/files/pft/pft-xl_step400k_ema.ckpt'
curl -fL --retry 3 -o checkpoints/sd_ae.ckpt \
  'https://www.dropbox.com/scl/fi/lvfvy7qou05kxfbqz5d42/sd_ae.ckpt?rlkey=fvtu2o48namouu9x3w08olv3o&st=vahu44z5&dl=1'
hf auth login      # DINOv3 là model gated: xin quyền truy cập trước
hf download facebook/dinov3-vits16-pretrain-lvd1689m --local-dir checkpoints/dinov3-vits16-pretrain-lvd1689m
```

- VAE phải là state dict `AutoencoderKL` của `jutils` mà [`vae.py`](vae.py) đọc. Một thư mục VAE của Diffusers không thay thế được.
- Từ V47 trở đi còn cần `checkpoints/garment_detail_scores.json`, dùng để oversample áo có chữ/logo. Tạo bằng lệnh:

```bash
PYTHONPATH=. python scripts/score_garment_detail.py \
  --root ../high-resolution-viton-zalando-dataset --output checkpoints/garment_detail_scores.json
```

Thư mục `checkpoints/` và `logs/` không nằm trong Git, nên phải tự sao lưu.

## 2. Dữ liệu VITON-HD và danh sách cặp

Tải [VITON-HD 1024 × 768 đã tiền xử lý](https://github.com/shadow2496/VITON-HD#dataset) và giải nén cạnh repo:

```text
../high-resolution-viton-zalando-dataset/
  train/{image,cloth,cloth-mask,agnostic-v3.2,agnostic-mask,image-densepose,image-parse-v3}/
  test/...   test_pairs.txt (unpaired)   test_pairs_paired.txt
```

**Cảnh báo:** `train_pairs.txt` trong một số bản ZIP gần như toàn cặp *unpaired*. Train trên file đó làm model học tái tạo một áo khác áo điều kiện. Tạo danh sách fit/dev paired, cùng định danh người–áo, giữ nguyên tập test chính thức:

```bash
python -m scripts.prepare_pfi_pairs --root ../high-resolution-viton-zalando-dataset \
  --seed 2026 --method stable-hash --dev-size 256
python -m scripts.prepare_pfi_pairs --root ../high-resolution-viton-zalando-dataset --check-only
```

- **Danh sách đúng:** fit 11.391 dòng (sha256 `2e6e879b…`), dev 256 dòng (`2c5bb414…`).
- **Danh sách lỗi** của sự cố V46: `fc4f2eab…` / `cf355ce9…`.
- **Bảo vệ khi resume:** checkpoint lưu fingerprint fit/dev, và trainer từ chối `--resume` nếu danh sách thay đổi.
- **Bảo vệ khi khởi tạo:** các nhánh ablation bật thêm `weights.init_pairs_must_match`, nên trainer cũng kiểm tra fingerprint khi khởi tạo từ checkpoint.

## 3. Train và resume

Mỗi version có một launcher. Launcher tự resume nếu `latest.pt` của run đó đã tồn tại:

```bash
bash scripts/run_pfi_v50_attn_flow.sh        # log: logs/vton-v50-attn-flow.log
```

Hoặc gọi trực tiếp:

```bash
PYTHONPATH=. python -m vton_ext.pfi_train --config configs/vton_v50_attn_flow.yaml --resume auto
```

- **Resume** (`--resume`): cần `latest.pt` đầy đủ, gồm model, optimizer và trạng thái curriculum, cùng kiến trúc và cùng danh sách cặp.
- **Khởi tạo trọng số** (`weights.init_from`): optimizer mới, và curriculum **bắt đầu lại từ đầu**. Muốn chạy tiếp một run đang dở curriculum thì phải dùng `--resume`.
- **Chạy thử ngắn:** thêm `train.stop_at_step=N`. Cách này không làm ngắn lịch LR của `train.max_steps`.
- **Override:** viết `key=value` sau đường dẫn config.
- **Bộ nhớ:** V50 ở 1024 × 768 dùng khoảng 38–40 GB VRAM và khoảng 21 s/bước trên RTX 4090 48 GB. `latest.pt` nặng khoảng 8 GB, và cần thêm khoảng 8 GB trống trong lúc lưu.

## 4. Ablation

Các nhánh ablation **không so với V50**, vì V50 có lịch sử dài và nhiều chặng. Thay vào đó, mọi nhánh khởi tạo từ cùng `logs/v46/step0001500.pt`:
- đây là V46 paired recovery ở bước 1.500, chỉ chứa trọng số, đã kiểm tra fingerprint sạch;
- mỗi nhánh chạy 5.000 bước với optimizer mới, cùng seed, cùng thứ tự dữ liệu và cùng lịch LR (warm-up 200, cosine).

Công thức chung của các nhánh:
- **Công thức V50 đầy đủ:** fine garment K/V từ block 21, attention flow, CoRAL, decoded RGB/HP, uncertainty, oversample chữ/logo.
- **Curriculum chạy lại từ đầu V46:** ramp nén còn 4.000 bước, rồi giữ hỗn hợp cuối thêm 1.000 bước.

So sánh E1–E4 với R0.

| Nhánh | Config | Thay đổi so với R0 |
| --- | --- | --- |
| R0 | `configs/ablation/r0_v50_recipe.yaml` | Không đổi gì (đối chứng) |
| E1 | `configs/ablation/e1_shared_timestep.yaml` | Mọi patch của một ảnh dùng chung một timestep (`flow.time_mode: shared`): lấy t của một token ngẫu nhiên trong ảnh, nên phân phối biên mỗi token giữ nguyên như R0 |
| E2 | `configs/ablation/e2_ltg_only.yaml` | Chỉ dùng LTG với tham số cố định (σ 0,6, loc 1,4, scale 1, shift 2), bỏ các nhánh noise/sync/detail và lịch thay đổi. Loc 1,4 làm t trung bình (0,346) khớp R0 tính trên cả quá trình train (0,344) |
| E3 | `configs/ablation/e3_full_attention.yaml` | Full attention (`model.attention_mode: full`): garment query đọc thêm K/V của person, tức hai chiều person ↔ garment. Không thêm tham số; garment K/V phụ thuộc trạng thái nhiễu nên không cache được khi suy luận |
| E4 | `configs/ablation/e4_no_decoded_loss.yaml` | Tắt decoded RGB và high-pass loss. Attention-flow loss (cũng tính trên ảnh) vẫn giữ |

```bash
bash scripts/run_ablation.sh r0 --bench     # 100 bước: chỉ đo tốc độ/bộ nhớ, thư mục output riêng
bash scripts/run_ablation.sh e3 --bench     # E3 chậm hơn; kiểm tra batch 4 có vừa không
bash scripts/run_ablation.sh r0             # train; tự resume logs/ablation/r0_v50_recipe/latest.pt
bash scripts/run_ablation.sh e3 train.batch_size=2 train.gradient_accumulation_steps=8   # vẫn batch hiệu dụng 16
```

- **Chuẩn bị:** máy train cần `logs/v46/step0001500.pt`, các checkpoint ở mục 1, và danh sách fit/dev paired sạch ở mục 2.
- **Chạy tuần tự:** mỗi lần chạy một nhánh, vì đĩa thường không đủ cho nhiều `latest.pt` cùng lúc.
- **Benchmark:** chỉ để chọn ngân sách, không dùng để xếp hạng chất lượng.

Những gì **không** ablate, và lý do:
- **Uncertainty loss:** sampler look-ahead cần log-variance dự đoán để chọn token khó, nên tắt loss này thì sampler mất ý nghĩa.
- **σ của LTG ≥ 0,5:** không có tác dụng. Độ lệch chuẩn thực tế là `min(t̄/2, σ)` và t̄ < 1, nên σ = 0,6 và σ = 0,9 sinh ra đúng cùng các timestep.

## 5. Đánh giá

### 5.1. Trên server (cách chính)

[`scripts/eval_pfi_test.sh`](../scripts/eval_pfi_test.sh) gọi [`scripts/eval_pfi_test.py`](../scripts/eval_pfi_test.py). Script đánh giá một checkpoint trên toàn bộ tập test VITON-HD: 2.032 cặp paired và 2.032 cặp unpaired. Cần GPU CUDA và các gói `lpips`, `torchmetrics`, `torch-fidelity` (`pip install lpips "torchmetrics[image]" torch-fidelity`).

```bash
# Chạy thử 16 mẫu trước để kiểm tra môi trường
LIMIT=16 bash scripts/eval_pfi_test.sh logs/vton-v50-attn-flow/latest.pt
# V50 đầy đủ: look-ahead 50 NFE, CFG 1,5, seed 42 (mặc định)
bash scripts/eval_pfi_test.sh logs/vton-v50-attn-flow/latest.pt
# Nhiều CFG trong một lần chạy; seed khác
CFG_SCALES=1,1.5,2 SEED=123 bash scripts/eval_pfi_test.sh logs/vton-v50-attn-flow/latest.pt
# Nhánh ablation: truyền config của nhánh làm tham số thứ hai; tham số sau đó chuyển thẳng cho file .py
bash scripts/eval_pfi_test.sh logs/ablation/e1_shared_timestep/latest.pt configs/ablation/e1_shared_timestep.yaml --sampler euler
# Lưu thêm ảnh ghép 4 cột cho mỗi mẫu; chỉ chạy paired
bash scripts/eval_pfi_test.sh logs/vton-v50-attn-flow/latest.pt configs/vton_v50_attn_flow.yaml --save-strips --settings pair
```

- **Tham số dòng lệnh:**
  - tham số đầu tiên là checkpoint;
  - tham số thứ hai (không bắt buộc) là config, mặc định `configs/vton_v50_attn_flow.yaml`;
  - các tham số còn lại chuyển thẳng cho `eval_pfi_test.py`, gồm `--sampler`, `--p`, `--context-ratio`, `--settings`, `--save-strips`, `--no-fid` và `--overwrite`.
- **Biến môi trường:** `DATA_ROOT` (mặc định `../high-resolution-viton-zalando-dataset`), `SAMPLER`, `NFE`, `CFG_SCALES`, `SEED`, `BATCH_SIZE`, `LIMIT`, `PY` (đường dẫn Python), `OUT` (thư mục kết quả).
- **Thư mục kết quả mặc định:** `outputs/eval/<tên run>-<sampler><NFE>-cfg<w>-seed<s>/`.
- **Danh sách paired:** paired nghĩa là mỗi người trong `test/image` mặc lại đúng áo của mình, tổng cộng 2.032 cặp. Một số bản tải về có `test_pairs_paired.txt` thiếu dòng; bản ở máy local chỉ có 1.953 dòng. Khi đó script tự tạo danh sách đầy đủ `test_pairs_paired_full.txt` trong thư mục kết quả, in cảnh báo và ghi số dòng vào `metadata.json`.

Kết quả lưu cho **từng mẫu**:

```text
<OUT>/metadata.json               checkpoint (đường dẫn, bước, kích thước), config, sampler, NFE, seed, p, context ratio, GPU
<OUT>/summary.json                tổng hợp mọi CFG và thiết lập
<OUT>/eval.log                    log
<OUT>/cfg<w>/pair/pred/<person>.png      ảnh dự đoán 1024 × 768 (đã ghép với ảnh agnostic)
<OUT>/cfg<w>/pair/strip/<person>.jpg     (với --save-strips) áo | agnostic | ảnh gốc / ground truth | dự đoán
<OUT>/cfg<w>/pair/per_sample.csv         index, person, garment, mse, mse_mask, ssim, lpips, sample_seconds
<OUT>/cfg<w>/pair/summary.json           trung bình MSE / MSE trong mặt nạ / SSIM / LPIPS, FID, KID, thời gian
<OUT>/cfg<w>/unpair/...                  như trên; per_sample.csv chỉ có tên ảnh và thời gian (không có ground truth)
```

Các thước đo:
- **MSE:** trên ảnh [0, 1]. `mse_mask` là MSE chỉ trong vùng mặt nạ edit.
- **SSIM:** cửa sổ Gauss 11, σ 1,5.
- **LPIPS:** AlexNet.
- **FID và KID:** tính bằng torch-fidelity, dùng chung đặc trưng Inception; so với `test/image`, tức ground truth ở paired và ảnh người thật ở unpaired. KID dùng 100 tập con × 1.000 ảnh và tự tắt khi ít hơn 1.000 mẫu.
- **Thời gian:** giây GPU cho mỗi ảnh, chỉ tính phần sampler, đã đồng bộ, bỏ batch đầu tiên.

Chạy tiếp và tái lập:
- **Nhiễu cố định theo mẫu:** nhiễu ban đầu của mỗi mẫu chỉ phụ thuộc `(seed, chỉ số mẫu)`, nên kết quả không đổi theo batch size.
- **Chạy tiếp:** chạy lại đúng lệnh cũ sẽ tiếp tục từ chỗ dừng.
- **Chống ghi đè nhầm:** nếu đổi checkpoint hoặc cấu hình mà vẫn dùng thư mục cũ, script dừng. Khi đó đổi `OUT` hoặc thêm `--overwrite`.
- **Dung lượng:** ảnh PNG 1024 × 768 khoảng 1–1,5 MB mỗi ảnh, tức khoảng 5–6 GB cho mỗi CFG (paired + unpaired).

Gộp mean ± std theo seed lấy mẫu, cho các thư mục nằm cùng một chỗ, ví dụ `outputs/eval/`:

```bash
for s in 42 123 2026; do SEED=$s bash scripts/eval_pfi_test.sh logs/vton-v50-attn-flow/latest.pt; done
python3 scripts/aggregate_pfi_eval.py --root outputs/eval
```

### 5.2. Trên Modal (tùy chọn)

[`docs/experiment/infer_modal.py`](../docs/experiment/infer_modal.py) dùng cùng thước đo, chạy trên H100 với volume `VTON-HD`. Script chỉ giữ 16 ảnh dự đoán mỗi cấu hình, nên muốn có ảnh và thước đo cho từng mẫu thì dùng mục 5.1. Mỗi cấu hình cần một `--output-tag` riêng.

```bash
modal run docs/experiment/infer_modal.py --nfe 50 --cfg-scales 1.5 --seed 42 --output-tag v50-la50-cfg1.5-seed42
modal run docs/experiment/infer_modal.py --sampler euler --nfe 50 --cfg-scales 1.5 --output-tag v50-euler50-cfg1.5-seed42
modal run docs/experiment/infer_modal.py --nfe 50 --cfg-scales 1.5 --p 0.5 --output-tag v50-la50-p05-cfg1.5-seed42
modal run docs/experiment/infer_modal.py --checkpoint /data/logs/ablation/e1_shared_timestep/latest.pt \
    --config-path configs/ablation/e1_shared_timestep.yaml --nfe 50 --cfg-scales 1.5 --output-tag e1-la50-cfg1.5-seed42
modal volume get VTON-HD experiments ./experiments && python3 scripts/aggregate_pfi_eval.py --root ./experiments
```

Lưu ý khi diễn giải kết quả:
- **Hai loại độ lệch chuẩn:** std theo seed lấy mẫu khác với std theo tập con của KID; không trộn hai loại.
- **Một seed train mỗi nhánh:** chênh lệch nhỏ hơn độ lệch theo seed lấy mẫu thì chưa đủ để kết luận.
- **Xem ảnh:** luôn xem ảnh vùng áo và logo bên cạnh các con số, vì FID/KID không cho biết chữ có đúng hay không. `per_sample.csv` giúp tìm nhanh các mẫu tệ nhất theo LPIPS.

## 6. Kiểm thử

```bash
PYTHONPATH=. /home/minh-le-vo-nhat/anaconda3/envs/vt-env/bin/python tests/test_pfi.py   # 42 test, CPU
```

Bộ test kiểm tra:
- mô hình ban đầu tương đương mô hình cha khi thêm nhánh mới;
- attention flow;
- timestep chung giữ đúng phân phối biên;
- trong full attention, garment đọc được trạng thái person và sampler vẫn chạy kèm CFG;
- cùng các bất biến khác của sampler và loss.
