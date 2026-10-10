# PFI-VTON: Patch Forcing Inpainting cho thử đồ ảo (VITON-HD 1024 × 768)

Đây là README chính của phần VTON. Mọi lệnh chạy từ **thư mục gốc của repo**.
Mô hình cuối là **V50** (`configs/vton_v50_attn_flow.yaml`), gồm:
- DiT PFT-XL/2 hai luồng (person / garment) với attention bất đối xứng;
- garment K/V mịn (8 px) ở các block cuối;
- các loss: CoRAL, decoded RGB/high-pass, uncertainty và attention-flow.

Ablation và đánh giá nằm ở mục 5–6. Các file đánh giá cụ thể ở [`docs/experiment/`](../docs/experiment/README.md).

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

**Cảnh báo:** `train_pairs.txt` trong một số bản ZIP gần như toàn cặp *unpaired*. Train trên file đó làm model học tái tạo một áo khác áo điều kiện; xem [biên bản sự cố V46](../docs/v46_resume_regression_audit_20260927.md). Tạo danh sách fit/dev paired, cùng định danh người–áo, giữ nguyên tập test chính thức:

```bash
python -m scripts.prepare_pfi_pairs --root ../high-resolution-viton-zalando-dataset \
  --seed 2026 --method stable-hash --dev-size 256
python -m scripts.prepare_pfi_pairs --root ../high-resolution-viton-zalando-dataset --check-only
```

- **Danh sách đúng:** fit 11.391 dòng (sha256 `2e6e879b…`), dev 256 dòng (`2c5bb414…`).
- **Danh sách lỗi** của sự cố V46: `fc4f2eab…` / `cf355ce9…`.
- **Bảo vệ khi resume:** checkpoint lưu fingerprint fit/dev, và trainer từ chối `--resume` nếu danh sách thay đổi.
- **Bảo vệ khi khởi tạo:** các nhánh ablation bật thêm `weights.init_pairs_must_match`, nên trainer cũng kiểm tra fingerprint khi khởi tạo từ checkpoint.

## 3. Lịch sử huấn luyện của V50

V50 là kết quả của nhiều chặng nối tiếp, không phải một lần train. Batch hiệu dụng luôn là 16 (4 ảnh × tích lũy 4). Mọi chặng dùng:
- AdamW (β = 0,9 / 0,999), weight decay 0;
- clip gradient 1,0; không dùng EMA;
- warm-up tuyến tính, sau đó cosine giảm còn 10%.

| Chặng | Độ phân giải | Khởi tạo | Số bước | Thay đổi chính | LR (mới / backbone) |
| --- | --- | --- | ---: | --- | --- |
| V40–V42 | 512 × 384 | PFT-XL/2 | 16.000 | Hai luồng, CoRAL; curriculum mở theo `coral_local_mass` ≥ 0,30; decoded RGB/HP từ V42 | 1e-4 / 2e-5, horizon 20k |
| V43 | 1024 × 768 | V42 bước 16.000 | 1.000 | Chuyển lên 1024 | 1e-5 / 5e-6 |
| V44 | 1024 × 768 | V43 bước 1.000 | 1.250 | Decoded loss trên toàn ảnh | 1e-5 / 5e-6 |
| V46 c2f | 1024 × 768 | V44 bước 1.250 | 1.500 | Curriculum thô → mịn theo bước | 4e-5 / 2e-5 |
| V46 recovery | 1024 × 768 | V46 c2f bước 1.500 | 8.250 | Chạy lại trên danh sách paired sạch | 4e-5 / 2e-5 |
| V47 | 1024 × 768 | V46 recovery bước 8.250 | 1.000 | Oversample áo có chữ/logo, CoRAL 0,3 | 2e-5 / 1e-5 |
| V48 | 1024 × 768 | V47 bước 1.000 | 5.500 | Fine garment K/V (residual trước LayerNorm, không học được) | 2e-5 / 1e-5, fine 1e-4 |
| V49 | 1024 × 768 | V48 bước 5.500 | 4.750 | Fine residual sau LayerNorm | như V48 |
| V50 | 1024 × 768 | V49 bước 4.750 | xem `checkpoint_step` | Attention-flow loss (Leffa) | như V48 |

Lấy số bước thật của V50 từ `metadata.json` của lần đánh giá (trường `checkpoint_step`). Không đoán con số này.

Tham số chung từ V46 trở đi:

| Nhóm | Giá trị |
| --- | --- |
| Time shift | 2 khi train, 1 khi lấy mẫu |
| LTG | σ = 0,6, scale 1 |
| Curriculum V46 | Bắt đầu: noise 0,35 / sync 0,25 / detail 0, loc −1. Kết thúc: 0,10 / 0,15 / 0,40, loc 0,8. Ramp tuyến tính 6.000 bước. Phần còn lại là LTG. Từ V47 giữ cố định hỗn hợp cuối |
| Nhánh detail | t̄ ∈ [0,75; 0,98], độ lệch chuẩn 0,05, không qua time shift |
| Trọng số loss | flow 1; uncertainty 0,01; CoRAL 0,3 (0,1 trước V47); entropy 0,01; RGB 0,5; HP 2,0 (vùng chữ/logo ×2, chỉ ảnh có t ≥ 0,65, 1 ảnh/microbatch); attention flow 0,25; spread 0,1 |
| Augment | Người: dịch 3%, scale 0,96–1,04. Áo: dịch 5%, scale 0,94–1,06 |
| Mặt nạ | Mở rộng 0–24 px khi train, 12 px khi suy luận; vùng mở rộng tô xám |
| CFG | Garment dropout 0,1 (áo và mặt nạ áo đặt về 0) |

## 4. Train và resume

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

## 5. Ablation

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

## 6. Đánh giá

[`docs/experiment/infer_modal.py`](../docs/experiment/infer_modal.py) chạy trên Modal (H100, volume `VTON-HD`). Mỗi job đánh giá một checkpoint với một cấu hình sampler, trên toàn bộ 2.032 cặp paired và unpaired của tập test. Script báo cáo:
- MSE, SSIM, LPIPS (AlexNet);
- FID và KID (torch-fidelity, cùng đặc trưng Inception; KID 100 tập con × 1.000 ảnh);
- thời gian lấy mẫu (giây/ảnh, đã đồng bộ GPU);
- số lượt gọi model và denoiser mỗi ảnh (CFG làm số lượt gọi denoiser tăng gấp đôi).

Mỗi cấu hình cần một `--output-tag` riêng.

```bash
# V50: độ lệch theo seed lấy mẫu (3 seed), CFG 1,5, look-ahead 50 NFE
for s in 42 123 2026; do modal run docs/experiment/infer_modal.py --nfe 50 --cfg-scales 1.5 --seed $s \
    --output-tag v50-la50-cfg1.5-seed$s; done
# V50: timestep riêng theo token (look-ahead) so với một timestep chung (Euler)
modal run docs/experiment/infer_modal.py --sampler euler --nfe 50 --cfg-scales 1.5 --output-tag v50-euler50-cfg1.5-seed42
# V50: tham số look-ahead (p = phân vị độ bất định; context ratio = khoảng nhìn trước)
modal run docs/experiment/infer_modal.py --nfe 50 --cfg-scales 1.5 --p 0.5 --output-tag v50-la50-p05-cfg1.5-seed42
modal run docs/experiment/infer_modal.py --nfe 50 --cfg-scales 1.5 --context-ratio 2.0 --output-tag v50-la50-r20-cfg1.5-seed42
# Một nhánh ablation (E1 chạy thêm --sampler euler)
modal run docs/experiment/infer_modal.py --checkpoint /data/logs/ablation/e1_shared_timestep/latest.pt \
    --config-path configs/ablation/e1_shared_timestep.yaml --nfe 50 --cfg-scales 1.5 --output-tag e1-la50-cfg1.5-seed42
# Gộp mean ± std theo seed, sau khi tải các file JSON về
modal volume get VTON-HD experiments ./experiments
python3 scripts/aggregate_pfi_eval.py --root ./experiments
```

Lưu ý khi diễn giải kết quả:
- **Hai loại độ lệch chuẩn:** std theo seed lấy mẫu khác với std theo tập con của KID; không trộn hai loại.
- **Một seed train mỗi nhánh:** chênh lệch nhỏ hơn độ lệch theo seed lấy mẫu thì chưa đủ để kết luận.
- **Xem ảnh crop:** luôn xem crop vùng áo và logo bên cạnh các con số, vì FID/KID không cho biết chữ có đúng hay không.

## 7. Kiểm thử

```bash
PYTHONPATH=. /home/minh-le-vo-nhat/anaconda3/envs/vt-env/bin/python tests/test_pfi.py   # 42 test, CPU
```

Bộ test kiểm tra:
- mô hình ban đầu tương đương mô hình cha khi thêm nhánh mới;
- attention flow;
- timestep chung giữ đúng phân phối biên;
- trong full attention, garment đọc được trạng thái person và sampler vẫn chạy kèm CFG;
- cùng các bất biến khác của sampler và loss.
