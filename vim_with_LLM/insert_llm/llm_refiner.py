import torch
import torch.nn as nn
from contextlib import nullcontext


class QwenLoRARefiner(nn.Module):
    """Refine per-timestep video features with a frozen Qwen backbone + LoRA adapters.

    Input:  x of shape [B, C, T]
    Output: refined x of shape [B, C, T]
    """

    def __init__(
        self,
        in_feat_dim: int,
        llm_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        alpha_init: float = 0.1,
        target_modules=None,
        torch_dtype: torch.dtype = torch.float16,
        max_note_tokens: int = 64,
        max_video_tokens: int = 192,
        gradient_checkpointing: bool = True,
        local_files_only: bool = False,
        device_map=None,
        max_memory=None,
    ):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for QwenLoRARefiner. "
                "Please install dependencies in vim/vim_requirements.txt"
            ) from e

        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "peft is required for LoRA. Install with `pip install peft`."
            ) from e

        self.in_feat_dim = in_feat_dim
        self.max_note_tokens = max_note_tokens
        self.max_video_tokens = max_video_tokens

        load_kwargs = dict(
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory

        base_model = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path,
            **load_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Use only the decoder backbone hidden states (not lm_head generation).
        self.backbone = base_model.model
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        for p in self.backbone.parameters():
            p.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.in_proj = nn.Linear(in_feat_dim, hidden_size)
        self.out_proj = nn.Linear(hidden_size, in_feat_dim)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=target_modules,
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        if gradient_checkpointing and hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

    def _forward_backbone_chunked(self, video_tokens, text_embeds=None, text_mask=None):
        # video_tokens: [1, T, H]
        # text_embeds:  [1, L, H] or None
        # text_mask:    [1, L] or None
        t = video_tokens.shape[1]
        chunk_size = int(self.max_video_tokens) if self.max_video_tokens is not None else t
        if chunk_size <= 0:
            chunk_size = t

        while True:
            try:
                outputs = []
                for start in range(0, t, chunk_size):
                    end = min(start + chunk_size, t)
                    seg_tokens = video_tokens[:, start:end, :]
                    if text_embeds is not None:
                        llm_inputs = torch.cat([text_embeds, seg_tokens], dim=1)
                        video_mask = torch.ones((1, seg_tokens.shape[1]), dtype=text_mask.dtype, device=seg_tokens.device)
                        attn_mask = torch.cat([text_mask, video_mask], dim=1)
                        y_all = self.backbone(inputs_embeds=llm_inputs, attention_mask=attn_mask).last_hidden_state
                        y_seg = y_all[:, text_embeds.shape[1]:, :]
                    else:
                        y_seg = self.backbone(inputs_embeds=seg_tokens).last_hidden_state
                    outputs.append(y_seg)
                return torch.cat(outputs, dim=1)
            except RuntimeError as e:
                oom = "out of memory" in str(e).lower()
                if (not oom) or chunk_size <= 1:
                    raise
                chunk_size = max(1, chunk_size // 2)
                if video_tokens.is_cuda:
                    torch.cuda.empty_cache()

    def _forward_backbone_chunked(self, video_tokens, text_embeds=None, text_mask=None):
        # video_tokens: [1, T, H]
        # text_embeds:  [1, L, H] or None
        # text_mask:    [1, L] or None
        t = video_tokens.shape[1]
        chunk_size = int(self.max_video_tokens) if self.max_video_tokens is not None else t
        if chunk_size <= 0:
            chunk_size = t

        outputs = []
        for start in range(0, t, chunk_size):
            end = min(start + chunk_size, t)
            seg_tokens = video_tokens[:, start:end, :]
            if text_embeds is not None:
                llm_inputs = torch.cat([text_embeds, seg_tokens], dim=1)
                video_mask = torch.ones((1, seg_tokens.shape[1]), dtype=text_mask.dtype, device=seg_tokens.device)
                attn_mask = torch.cat([text_mask, video_mask], dim=1)
                y_all = self.backbone(inputs_embeds=llm_inputs, attention_mask=attn_mask).last_hidden_state
                y_seg = y_all[:, text_embeds.shape[1]:, :]
            else:
                y_seg = self.backbone(inputs_embeds=seg_tokens).last_hidden_state
            outputs.append(y_seg)

        return torch.cat(outputs, dim=1)

    def _encode_text(self, text: str, device, dtype):
        if text is None:
            text = ""
        text = str(text)
        tok = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.max_note_tokens,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].to(device=device, dtype=torch.long)
        text_mask = tok["attention_mask"].to(device=device, dtype=torch.long)
        text_embeds = self.backbone.embed_tokens(input_ids).to(dtype)
        return text_embeds, text_mask

    def _project_out(self, x: torch.Tensor) -> torch.Tensor:
        """Device-safe projection for model-parallel outputs."""
        weight = self.out_proj.weight
        bias = self.out_proj.bias
        if weight.device != x.device or weight.dtype != x.dtype:
            weight = weight.to(device=x.device, dtype=x.dtype)
            if bias is not None:
                bias = bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)

    def forward(self, x: torch.Tensor, notes=None, segment_notes=None, segment_spans=None) -> torch.Tensor:
        # x: [B, C, T]
        embed_tokens = getattr(self.backbone, "embed_tokens", None)
        target_device = embed_tokens.weight.device if embed_tokens is not None else self.in_proj.weight.device
        target_dtype = embed_tokens.weight.dtype if embed_tokens is not None else self.in_proj.weight.dtype

        if self.in_proj.weight.device != target_device or self.in_proj.weight.dtype != target_dtype:
            self.in_proj = self.in_proj.to(device=target_device, dtype=target_dtype)
            self.out_proj = self.out_proj.to(device=target_device, dtype=target_dtype)
            self.alpha.data = self.alpha.data.to(target_device)

        if x.device != target_device:
            x = x.to(target_device)

        x_bt = x.transpose(1, 2)  # [B, T, C]
        x_bt_model = x_bt.to(dtype=target_dtype)
        h = self.in_proj(x_bt_model)

        # Segment-aligned mode: refine each temporal span with its paired segment text.
        if segment_notes is not None and segment_spans is not None:
            refined_h = h.clone()
            batch_size = x_bt.shape[0]
            for b in range(batch_size):
                sample_notes = segment_notes[b] if b < len(segment_notes) else []
                sample_spans = segment_spans[b] if b < len(segment_spans) else []
                n_seg = min(len(sample_notes), len(sample_spans))
                for i in range(n_seg):
                    s, e = sample_spans[i]
                    s = max(0, int(s))
                    e = min(int(e), h.shape[1])
                    if e <= s:
                        continue
                    seg_tokens = h[b:b+1, s:e, :]  # [1, t_seg, H]
                    text_embeds, text_mask = self._encode_text(sample_notes[i], h.device, target_dtype)
                    llm_inputs = torch.cat([text_embeds, seg_tokens], dim=1)
                    video_mask = torch.ones((1, seg_tokens.shape[1]), dtype=text_mask.dtype, device=h.device)
                    attn_mask = torch.cat([text_mask, video_mask], dim=1)
                    y_all = self.backbone(inputs_embeds=llm_inputs, attention_mask=attn_mask).last_hidden_state
                    text_len = text_embeds.shape[1]
                    y_seg = y_all[:, text_len:, :]
                    delta_seg = self.out_proj(y_seg)
                    if delta_seg.device != seg_tokens.device:
                        delta_seg = delta_seg.to(seg_tokens.device)
                    refined_h[b:b+1, s:e, :] = seg_tokens + self.alpha.to(device=seg_tokens.device, dtype=seg_tokens.dtype) * delta_seg

            return refined_h.to(dtype=x_bt.dtype).transpose(1, 2)  # [B, C, T]

        if notes is not None:
            if isinstance(notes, str):
                notes = [notes]
            notes = ["" if n is None else str(n) for n in list(notes)]
            if len(notes) != x_bt.shape[0]:
                raise ValueError(f"notes size mismatch: got {len(notes)} notes for batch {x_bt.shape[0]}")
            tok = self.tokenizer(notes, padding=True, truncation=True, max_length=self.max_note_tokens, return_tensors="pt")
            input_ids = tok["input_ids"].to(device=x_bt.device, dtype=torch.long)
            text_mask = tok["attention_mask"].to(device=x_bt.device, dtype=torch.long)
            text_embeds = self.backbone.embed_tokens(input_ids).to(dtype=target_dtype)
            llm_inputs = torch.cat([text_embeds, h], dim=1)
            video_mask = torch.ones((x_bt.shape[0], h.shape[1]), dtype=text_mask.dtype, device=x_bt.device)
            attn_mask = torch.cat([text_mask, video_mask], dim=1)
            y_all = self.backbone(inputs_embeds=llm_inputs, attention_mask=attn_mask).last_hidden_state
            text_len = text_embeds.shape[1]
            y = y_all[:, text_len:, :]
        else:
            y_list = []
            for b in range(x_bt.shape[0]):
                y_b = self._forward_backbone_chunked(h[b:b+1, :, :])
                y_list.append(y_b)
            y = torch.cat(y_list, dim=0)
        delta = self.out_proj(y)
        if delta.device != x_bt_model.device:
            delta = delta.to(x_bt_model.device)

        x_refined = x_bt_model + self.alpha.to(x_bt_model.dtype) * delta
        x_refined = x_refined.to(dtype=x_bt.dtype)
        return x_refined.transpose(1, 2)  # [B, C, T]


@torch.no_grad()
def refine_feature_tensor(
    model: QwenLoRARefiner,
    feats: torch.Tensor,
    device: torch.device,
    chunk_size: int = 512,
    note: str = "",
) -> torch.Tensor:
    """Refine a single video feature tensor.

    Args:
        feats: [T, C] float32/float16 feature tensor
        chunk_size: number of timesteps per forward pass to control memory

    Returns:
        [T, C] refined feature tensor
    """
    model.eval()
    outputs = []

    for start in range(0, feats.shape[0], chunk_size):
        end = min(start + chunk_size, feats.shape[0])
        chunk = feats[start:end].to(device)
        chunk = chunk.transpose(0, 1).unsqueeze(0)  # [1, C, t]
        refined = model(chunk, notes=[note]).squeeze(0).transpose(0, 1)  # [t, C]
        outputs.append(refined.detach().cpu())

    return torch.cat(outputs, dim=0)
