/* Shape atlas for the implementation in vton_ext. No checkpoint is loaded. */
(() => {
  'use strict';

  const els = {
    resolution: document.getElementById('resolution'),
    batch: document.getElementById('batch'),
    sampler: document.getElementById('sampler'),
    nfe: document.getElementById('nfe'),
    cfg: document.getElementById('cfg'),
    tokens: document.getElementById('metric-tokens'),
    grid: document.getElementById('metric-grid'),
    keys: document.getElementById('metric-keys'),
    run: document.getElementById('metric-run'),
    runLabel: document.getElementById('metric-run-label'),
    runNote: document.getElementById('metric-run-note'),
    positionGrid: document.getElementById('pos-grid'),
    samplerTitle: document.getElementById('sampler-title'),
    recipeHint: document.getElementById('recipe-hint'),
    lossTitle: document.getElementById('loss-title'),
    lossDesc: document.getElementById('loss-desc'),
    curriculumProgress: document.getElementById('curriculum-progress'),
    curriculumNote: document.getElementById('curriculum-note'),
    curriculumPercent: document.getElementById('curriculum-percent'),
    inspectorIndex: document.getElementById('inspector-index'),
    inspectorGlyph: document.getElementById('inspector-glyph'),
    inspectorTag: document.getElementById('inspector-tag'),
    inspectorTitle: document.getElementById('inspector-title'),
    inspectorDesc: document.getElementById('inspector-desc'),
    inspectorInput: document.getElementById('inspector-input'),
    inspectorOutput: document.getElementById('inspector-output'),
    inspectorDetail: document.getElementById('inspector-detail'),
    inspectorFormula: document.getElementById('inspector-formula'),
    inspectorSource: document.getElementById('inspector-source'),
  };
  const nodes = [...document.querySelectorAll('[data-node]')];
  const shapeFields = [...document.querySelectorAll('[data-shape]')];
  let selected = 'personTokens';

  function context() {
    const height = Number(els.resolution.value);
    const width = height * 3 / 4;
    const b = Number(els.batch.value);
    const latentH = height / 8;
    const latentW = width / 8;
    const tokenH = latentH / 2;
    const tokenW = latentW / 2;
    const n = tokenH * tokenW;
    return {
      height, width, b, latentH, latentW, tokenH, tokenW, n,
      nfe: Number(els.nfe.value),
      cfg: Number(els.cfg.value),
      sampler: els.sampler.value,
      train: document.querySelector('input[name="mode"]:checked').value === 'train',
    };
  }

  function shapes(c) {
    const {b, height:h, width:w, latentH:lh, latentW:lw, n} = c;
    const latent = (channels) => `[${b}, ${channels}, ${lh}, ${lw}]`;
    const rgb = (channels) => `[${b}, ${channels}, ${h}, ${w}]`;
    return {
      personInput: `RGB ${rgb(3)} · M ${rgb(1)}`,
      poseInput: rgb(3),
      poseLatent: latent(4),
      garmentInput: `RGB ${rgb(3)} · M ${rgb(1)}`,
      garmentEncoded: `z_G ${latent(4)} + m_G ${latent(1)}`,
      known: latent(4),
      cond: latent(9),
      garmentLatent: latent(5),
      personTokens: `[${b}, ${n}, 1152]`,
      garmentTokens: `[${b}, ${n}, 1152]`,
      position: `[1, ${n}, 1152]`,
      timeMask: `edit bool [${b},${n}] · t float [${b},${n}]`,
      personAttention: `Q [${b},16,${n},72] → K/V [${b},16,${2*n},72]`,
      garmentAttention: `Q/K/V [${b},16,${n},72]`,
      kvCache: `28 × 2 × [${b},16,${n},72]`,
      coral: `4 blocks × [${b},4,${n},${n}]`,
      head: `[${b},${n},20] → v ${latent(4)} + s ${latent(1)}`,
      samplerNode: `${latent(4)} · ${c.nfe} NFE`,
      result: rgb(3),
      target: `RGB ${rgb(3)} → x₁ ${latent(4)}`,
      targetLatentAndU: `x₁ ${latent(4)} + u ${latent(4)}`,
      dino: `similarity [${b},${n},${n}]`,
      loss: 'scalar []',
    };
  }

  const info = {
    personInput: {
      glyph:'◧', tag:'VTON INPUT', title:'Người agnostic + mask',
      desc:'Ảnh người đã bỏ thông tin áo cũ và mask xác định vùng sinh mới.',
      input:'Ảnh agnostic + mask gốc', out:'personInput',
      detail:'Mask được giãn bằng max-pool. Phần nằm trong mask được fill 0 ở thang RGB [-1,1]. Mask quyết định token nào EDIT; các pixel quan sát được ngoài mask sẽ được giữ lại ở cuối.',
      formula:'A_masked = A · (1 − M_open)', source:'../vton_ext/pfi_sample.py',
    },
    poseInput: {
      glyph:'⌁', tag:'VTON INPUT', title:'DensePose',
      desc:'Biểu diễn tư thế được VAE encode thành điều kiện không gian.',
      input:'poseInput', out:'poseLatent',
      detail:'DensePose đi vào frozen VAE encoder, tạo latent 4 kênh. Nó được ghép với agnostic latent và mask để thành cond 9 kênh, không đi qua một transformer riêng.',
      formula:'z_P = E(P)', source:'../vton_ext/pfi_sample.py',
    },
    garmentInput: {
      glyph:'▧', tag:'VTON INPUT', title:'Áo tham chiếu + mask',
      desc:'Ảnh áo shop và mask áo dùng làm điều kiện sạch suốt trajectory.',
      input:'garmentInput', out:'garmentEncoded',
      detail:'Ảnh áo đi qua frozen VAE encoder; mask áo được average-pool xuống lưới latent. Training có garment dropout 10% trong config V45 để hỗ trợ CFG.',
      formula:'z_G = E(G); m_G = AvgPool(M_G)', source:'../vton_ext/pfi_sample.py',
    },
    known: {
      glyph:'▤', tag:'FROZEN VAE', title:'Agnostic latent / known context',
      desc:'Latent sạch của vùng người được quan sát, giữ cố định trong sampling.',
      input:'personInput', out:'known',
      detail:'VAE giảm kích thước 8 lần trên mỗi chiều. Ở cả train và inference, các patch ngoài edit mask lấy từ agnostic latent này và có t=1, nên train/test dùng cùng dạng context.',
      formula:'known = E(A_masked)', source:'../vton_ext/pfi_sample.py',
    },
    cond: {
      glyph:'⊕', tag:'NEW PROJECTION', title:'Điều kiện VTON 9 kênh',
      desc:'Agnostic latent, mask latent và DensePose latent được nối theo kênh.',
      input:'known + mask [B,1,h,w] + pose [B,4,h,w]', out:'cond',
      detail:'Conv2d 9→1152, kernel/stride 2, khởi tạo toàn zero. Output được cộng vào person patch embedding gốc, vì thế nhánh này lúc mới khởi tạo chưa làm đổi đặc trưng PFT.',
      formula:'cond = cat(z_A[4], m[1], z_P[4])', source:'../vton_ext/pfi_model.py',
    },
    garmentLatent: {
      glyph:'▥', tag:'NEW PROJECTION', title:'Garment latent 5 kênh',
      desc:'Latent áo 4 kênh và mask áo 1 kênh được patchify cùng nhau.',
      input:'garmentInput', out:'garmentLatent',
      detail:'Conv2d 5→1152, kernel/stride 2. Bốn kênh latent khởi tạo bằng weight patch embed ảnh của PFT; kênh mask zero. Garment tokens đều t=1.',
      formula:'garment_in = cat(z_G[4], m_G[1])', source:'../vton_ext/pfi_model.py',
    },
    personTokens: {
      glyph:'▦', tag:'PFT WEIGHTS + VTON COND', title:'Person tokens',
      desc:'Patch edit bắt đầu từ noise; patch known luôn sạch. Mỗi token mang thời gian và role riêng.',
      input:'x_t [B,4,h,w] + cond [B,9,h,w]', out:'personTokens',
      detail:'Dùng x_embedder gốc (4→1152) rồi cộng cond_embedder mới, vị trí chữ nhật và role EDIT/KNOWN. AdaLN condition = TimeEmbed(t_i) + pretrained null-class + role_cond. Toàn bộ backbone được fine-tune.',
      formula:'h_p = Patch(x_t) + Patch(cond) + pos + role', source:'../vton_ext/pfi_model.py',
    },
    garmentTokens: {
      glyph:'▦', tag:'SHARED BACKBONE', title:'Garment tokens',
      desc:'Áo sạch được đưa vào cùng 28 block PFT, không có ReferenceNet độc lập.',
      input:'garmentLatent', out:'garmentTokens',
      detail:'Tất cả token garment có t=1 và role GARMENT. Chúng tự attention trong mỗi block, không đọc person. Do đó trạng thái và K/V của áo không đổi theo từng bước denoise.',
      formula:'h_g = Patch([z_G,m_G]) + pos + role_G', source:'../vton_ext/pfi_model.py',
    },
    position: {
      glyph:'⌗', tag:'RESOLUTION ADAPTATION', title:'Vị trí chữ nhật',
      desc:'Bảng vị trí pretrained vuông 16×16 được nội suy cho lưới token hiện tại.',
      input:'PFT pos [1,256,1152]', out:'position',
      detail:'Nội suy bicubic giữ tương thích tọa độ tương đối khi chuyển độ phân giải. V40–V42 giữ bảng cố định; V45 bật pos_embed_trainable=true, cho phép điều chỉnh thêm 3.538.944 giá trị ở 1024×768.',
      formula:'pos = Bicubic(pos_PFT_16×16 → grid)', source:'../vton_ext/utils.py',
    },
    timeMask: {
      glyph:'◷', tag:'PFT TIME + VTON CURRICULUM', title:'Thời gian và mask theo patch',
      desc:'Mỗi token EDIT có thời gian riêng; context và áo luôn sạch ở t=1.',
      input:'mask latent [B,1,h,w] + time sampler', out:'timeMask',
      detail:'Ở inference, EDIT bắt đầu t=0 và được tích phân đến 1; KNOWN và garment giữ t=1. Training dùng hỗn hợp pure noise, synchronous, detail và LTG. Pure noise giúp model học đọc áo; detail tăng cơ hội học logo và cấu trúc tần số cao.',
      formula:'t_i = t_edit,i nếu EDIT; ngược lại t_i=1', source:'../vton_ext/pfi_train.py',
    },
    personAttention: {
      glyph:'↘', tag:'PFT QKV · VTON ROUTING', title:'Person attention',
      desc:'Query của người đọc cả người lẫn áo trong cùng một SDPA.',
      input:'personTokens + kvCache', out:'personTokens',
      detail:'Projection Q/K/V, q/k normalization, adaLN, MLP là weight của PFT. Khác biệt là K và V của person được nối với garment K/V. Shape N×2N là ma trận attention logic; SDPA không nhất thiết materialize toàn bộ.',
      formula:'SDPA(Q_p, [K_p;K_g], [V_p;V_g])', source:'../vton_ext/pfi_model.py',
    },
    garmentAttention: {
      glyph:'⟲', tag:'SHARED WEIGHTS · NEW PATH', title:'Garment self-attention',
      desc:'Áo chạy trước người qua chính block PFT đó, nhưng chỉ đọc chính áo.',
      input:'garmentTokens', out:'garmentTokens',
      detail:'Ở block ℓ, garment dùng cùng norm, QKV, attention, adaLN và MLP như person. Garment query không bao giờ đọc person. Điều này tạo điều kiện cho cache chính xác ở inference.',
      formula:'SDPA(Q_g, K_g, V_g) → h_g^(ℓ+1)', source:'../vton_ext/pfi_model.py',
    },
    kvCache: {
      glyph:'⇢', tag:'EXACT INFERENCE CACHE', title:'28 cặp garment K/V',
      desc:'Một cặp key/value cho mỗi block; person tái sử dụng ở mọi NFE.',
      input:'garmentTokens', out:'kvCache',
      detail:'encode_garment tính một lần cho áo thật. Với garment CFG, model tính thêm một cache áo rỗng, rồi mỗi bước chạy person hai lần. Cache không xấp xỉ: garment branch hoàn toàn độc lập với person state.',
      formula:'cache[ℓ] = (K_g^ℓ, V_g^ℓ), ℓ=0…27', source:'../vton_ext/pfi_model.py',
    },
    coral: {
      glyph:'◎', tag:'TRAIN ONLY', title:'CoRAL attention readout',
      desc:'Giám sát hướng đọc áo bằng correspondence từ DINOv3.',
      input:'Q_p + K_g ở block 8,12,16,20', out:'coral',
      detail:'Chỉ bốn head đầu của mỗi block được giám sát. Map này softmax trên garment keys, trong khi attention thật softmax trên cả person + garment keys; vì vậy nó chỉ cho biết vị trí trên áo, không cho biết tổng attention mass vào áo.',
      formula:'A_g = softmax(Q_p K_gᵀ / √72)', source:'../vton_ext/coral.py',
    },
    head: {
      glyph:'↧', tag:'PFT FINAL LAYER', title:'Velocity + uncertainty',
      desc:'Final layer gốc áp dụng lên person tokens rồi unpatchify.',
      input:'personTokens', out:'head',
      detail:'Mỗi token sinh 2×2×5 = 20 số. Kênh 0…3 là velocity flow; kênh 4 là log-variance dùng cho NLL khi train và xếp hạng độ khó patch khi sampling.',
      formula:'Final(h_p) → Unpatchify → (v[4], logvar[1])', source:'../vton_ext/pfi_model.py',
    },
    samplerNode: {
      glyph:'◴', tag:'VTON SAMPLING', title:'Masked patch sampling',
      desc:'Chỉ vùng EDIT được tích phân; KNOWN luôn giữ agnostic latent sạch.',
      input:'head', out:'samplerNode',
      detail:'Dual-loop chia bước nhỏ cho khoảng 30% token EDIT bất định nhất khi p=0.7; mọi denoiser call vẫn xử lý toàn bộ person sequence. Euler đi đều; look-ahead cho easy patches chạy trước làm context. NFE đếm evaluation, còn CFG ≠ 1 tăng gấp đôi person calls.',
      formula:'x ← x + Δt_i · v, chỉ khi edit_i=true', source:'../vton_ext/pfi_sample.py',
    },
    result: {
      glyph:'◩', tag:'FROZEN VAE + COMPOSITE', title:'Ảnh try-on hoàn chỉnh',
      desc:'Giải mã latent và giữ nguyên người quan sát được ngoài mask.',
      input:'samplerNode', out:'result',
      detail:'Frozen VAE decoder sinh ảnh RGB đầy đủ. Composite lấy ảnh decode ở trong mask, agnostic RGB ở ngoài mask. V45 eval bật seam correction để giảm chênh màu ở biên.',
      formula:'I_out = M·D(x_final) + (1−M)·A', source:'../vton_ext/pfi_sample.py',
    },
    target: {
      glyph:'◈', tag:'TRAIN ONLY', title:'Target và flow target',
      desc:'Ảnh người mặc đúng áo chỉ xuất hiện trong đường training.',
      input:'target', out:'targetLatentAndU',
      detail:'Frozen VAE mã hóa ảnh target thành x₁. Nhiễu x₀∼N(0,I), x_t=t·x₁+(1−t)·x₀ tại patch EDIT; known patch thay bằng agnostic latent. Velocity mục tiêu u=x₁−x₀. Không có target trong prepare_inputs hoặc inference.',
      formula:'x_t=t·x₁+(1−t)·ε; u=x₁−ε', source:'../vton_ext/pfi_train.py',
    },
    dino: {
      glyph:'✳', tag:'FROZEN TEACHER', title:'DINOv3 correspondence',
      desc:'Teacher đóng băng so khớp target người với ảnh áo trong train.',
      input:'target + garmentInput', out:'dino',
      detail:'Feature DINOv3-S/16 của ảnh người và áo được đưa về token grid, cosine similarity tạo ma trận N×N. Chỉ match đủ giống và cycle-consistent được dùng làm target CoRAL. Teacher không tham gia inference.',
      formula:'S = F_person · F_garmentᵀ', source:'../vton_ext/coral.py',
    },
    loss: {
      glyph:'∑', tag:'VTON OBJECTIVES', title:'Các loss của VTON',
      desc:'Flow, uncertainty, correspondence và decoded detail cùng tối ưu backbone.',
      input:'head + coral + dino + target', out:'loss',
      detail:'Flow MSE tính ở vùng edit, tăng trọng số ở áo. NLL học log-variance. CoRAL CE/entropy định tuyến head chọn lọc. V45 chọn một phần mẫu thời gian gần sạch để decode endpoint và tính RGB/high-pass; VAE frozen nhưng gradient vẫn đi qua decoder về DiT.',
      formula:'L = L_flow + λ_NLL L_NLL + λ_C L_CoRAL + L_RGB/detail', source:'../vton_ext/pfi_train.py',
    },
  };

  function resolveShape(value, sh) { return sh[value] || value; }

  function renderInspector(c, sh) {
    const selectedNode = info[selected];
    const visible = nodes.filter(node => getComputedStyle(node).display !== 'none');
    const current = visible.findIndex(node => node.dataset.node === selected);
    els.inspectorIndex.textContent = `${String(current + 1).padStart(2,'0')} / ${String(visible.length).padStart(2,'0')}`;
    els.inspectorGlyph.textContent = selectedNode.glyph;
    els.inspectorTag.textContent = selectedNode.tag;
    els.inspectorTitle.textContent = selectedNode.title;
    els.inspectorDesc.textContent = selected === 'loss' && c.height === 512
      ? 'Flow, uncertainty và correspondence của recipe V40.'
      : selectedNode.desc;
    els.inspectorInput.textContent = resolveShape(selectedNode.input, sh);
    els.inspectorOutput.textContent = resolveShape(selectedNode.out, sh);
    let detail = selectedNode.detail;
    if (selected === 'position') {
      detail = c.height === 1024
        ? 'PFT 16×16 được nội suy bicubic sang 64×48. V45 bật pos_embed_trainable=true: 3.538.944 giá trị vị trí được fine-tune từ bảng đã nội suy.'
        : 'PFT 16×16 được nội suy bicubic sang 32×24. V40–V42 dùng bảng vị trí cố định trong quá trình training.';
    }
    if (selected === 'samplerNode') {
      const calls = c.nfe * (c.cfg === 1 ? 1 : 2);
      detail = `${selectedNode.detail} Thiết lập đang chọn: ${c.sampler}, ${c.nfe} NFE, CFG ${c.cfg}; ${calls} person calls và ${c.cfg === 1 ? 1 : 2} garment cache(s).`;
    }
    if (selected === 'timeMask') {
      detail = c.train
        ? (c.height === 512
            ? 'V40 dùng CoRAL gate rồi ramp từ 50/20/0/30 sang 10/15/40/35% cho pure noise/synchronous/detail/LTG. Thanh kéo bên dưới hiển thị tiến độ ramp sau khi gate mở; từng ảnh chỉ chọn một nhánh, còn LTG có thời gian khác nhau theo patch.'
            : 'V45 fine-tune ở 1024×768 dùng hỗn hợp 10/15/40/35% ngay từ đầu. time_shift=2 cho các nhánh thường; detail_unshifted=true giữ khoảng thời gian detail gần sạch. Mỗi ảnh chọn một nhánh, LTG tạo thời gian riêng cho patch.')
        : 'Ở inference, EDIT bắt đầu t=0 và tiến tới t=1; KNOWN và garment giữ t=1. Chỉ token EDIT được cập nhật latent. Sampler có thể tạo các t_i khác nhau cho hard/easy token.';
    }
    if (selected === 'loss' && c.height === 512) {
      detail = 'V40 tối ưu flow MSE trên vùng edit, NLL từ log-variance và CoRAL CE/entropy ở bốn block chọn lọc. Loss decoded RGB/high-pass chỉ được thêm ở các recipe V42 về sau và V43–V45.';
    }
    els.inspectorDetail.textContent = detail;
    els.inspectorFormula.textContent = selected === 'loss' && c.height === 512
      ? 'L = L_flow + λ_NLL L_NLL + λ_C L_CoRAL'
      : selectedNode.formula;
    els.inspectorSource.href = selectedNode.source;
    nodes.forEach(node => node.setAttribute('aria-pressed', String(node.dataset.node === selected)));
  }

  function render() {
    const c = context();
    const sh = shapes(c);
    document.body.classList.toggle('mode-train', c.train);
    if (c.train && (selected === 'samplerNode' || selected === 'result')) selected = 'personTokens';
    if (!c.train && (selected === 'coral' || selected === 'target' || selected === 'dino' || selected === 'loss')) selected = 'personTokens';
    shapeFields.forEach(el => { el.textContent = sh[el.dataset.shape]; });
    const fmt = number => new Intl.NumberFormat('vi-VN').format(number);
    els.tokens.textContent = fmt(c.n);
    els.grid.textContent = `${c.tokenH} × ${c.tokenW} grid`;
    els.keys.textContent = `${fmt(c.n)} → ${fmt(2*c.n)}`;
    els.positionGrid.textContent = `${c.tokenH}×${c.tokenW}`;
    els.samplerTitle.textContent = ({dual_loop:'Dual-loop · masked',euler:'Euler · masked',look_ahead:'Look-ahead · masked'})[c.sampler];
    els.recipeHint.textContent = `Shape [B, C, H, W] · recipe ${c.height === 1024 ? 'V45' : 'V40'}`;
    els.lossTitle.textContent = c.height === 1024 ? 'Flow + NLL + CoRAL + RGB' : 'Flow + NLL + CoRAL';
    els.lossDesc.textContent = c.height === 1024 ? 'decoded RGB/high-pass trên mẫu chọn (V45)' : 'V40: chưa có decoded detail loss';
    const v40 = c.height === 512;
    els.curriculumProgress.disabled = !v40;
    if (!v40) els.curriculumProgress.value = '100';
    const progress = Number(els.curriculumProgress.value) / 100;
    const start = [0.50, 0.20, 0, 0.30];
    const end = [0.10, 0.15, 0.40, 0.35];
    const probabilities = start.map((value, index) => v40 ? value + (end[index] - value) * progress : end[index]);
    ['noise','sync','detail','ltg'].forEach((key, index) => {
      const percent = (probabilities[index] * 100).toFixed(1).replace(/\.0$/, '');
      document.getElementById(`bar-${key}`).style.width = `${probabilities[index] * 100}%`;
      document.getElementById(`pct-${key}`).textContent = `${percent}%`;
    });
    els.curriculumNote.textContent = v40 ? 'V40 · ramp sau CoRAL gate' : 'V45 · fine-tune từ hỗn hợp cuối';
    els.curriculumPercent.textContent = v40 ? `${els.curriculumProgress.value}%` : 'cố định';
    if (c.train) {
      els.runLabel.textContent = 'TRAINING OBJECTIVES';
      els.run.textContent = c.height === 1024 ? '4 loss groups' : '3 loss groups';
      els.runNote.textContent = c.height === 1024 ? 'Flow · NLL · CoRAL · decoded detail (V45)' : 'Flow · NLL · CoRAL (V40)';
    } else {
      els.runLabel.textContent = 'PERSON CALLS / GARMENT CACHES';
      els.run.textContent = `${c.nfe * (c.cfg === 1 ? 1 : 2)} / ${c.cfg === 1 ? 1 : 2}`;
      els.runNote.textContent = `${c.nfe} NFE · CFG ${c.cfg} · ${c.sampler.replace('_','-')}`;
    }
    renderInspector(c, sh);
  }

  // Query parameters make a specific view reproducible when sharing the file/URL.
  const params = new URLSearchParams(window.location.search);
  for (const [key, element] of Object.entries({resolution:els.resolution,batch:els.batch,sampler:els.sampler,nfe:els.nfe,cfg:els.cfg})) {
    const value = params.get(key);
    if (value && [...element.options].some(option => option.value === value)) element.value = value;
  }
  const mode = params.get('mode');
  if (mode === 'train' || mode === 'infer') document.querySelector(`input[name="mode"][value="${mode}"]`).checked = true;
  const curriculumProgress = params.get('progress');
  if (curriculumProgress !== null && /^\d+$/.test(curriculumProgress) && Number(curriculumProgress) <= 100) {
    els.curriculumProgress.value = curriculumProgress;
  }

  nodes.forEach(node => node.addEventListener('click', () => { selected = node.dataset.node; render(); }));
  [els.resolution, els.batch, els.sampler, els.nfe, els.cfg, ...document.querySelectorAll('input[name="mode"]')]
    .forEach(input => input.addEventListener('change', render));
  els.curriculumProgress.addEventListener('input', render);
  render();
})();
