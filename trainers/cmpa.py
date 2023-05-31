import os.path as osp
from collections import OrderedDict
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CMPA',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "cmpa_length": cfg.TRAINER.CMPA.N_CTX,
                      "fusing": cfg.TRAINER.CMPA.FUSING,
                      "parameter_sharing": cfg.TRAINER.CMPA.PS}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class Encoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # visual
        self.conv1_visual = clip_model.conv1_visual
        self.class_embedding_visual = clip_model.class_embedding_visual
        self.positional_embedding_visual = clip_model.positional_embedding_visual
        self.ln_pre_visual = clip_model.ln_pre_visual
        self.ln_post_visual = clip_model.ln_post_visual
        self.proj_visual = clip_model.proj_visual
        
        # visual and language attention
        self.resblocks = clip_model.resblocks
        
        # language
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x_visual, visual_ctx, prompts_text, tokenized_prompts, compound_prompts_depth):
        # visual pre-pro
        x_visual = self.conv1_visual(x_visual)  # shape = [*, width, grid, grid]
        x_visual = x_visual.reshape(x_visual.shape[0], x_visual.shape[1], -1)  # shape = [*, width, grid ** 2]
        x_visual = x_visual.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x_visual = torch.cat(
            [self.class_embedding_visual.to(x_visual.dtype) + torch.zeros(x_visual.shape[0], 1, x_visual.shape[-1], dtype=x_visual.dtype, device=x_visual.device),
             x_visual], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x_visual = x_visual + self.positional_embedding_visual.to(x_visual.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        visual_ctx = visual_ctx.expand(x_visual.shape[0], -1, -1)
        x_visual = torch.cat([x_visual, visual_ctx], dim=1)

        # Normal code as before
        x_visual = self.ln_pre_visual(x_visual)

        x_visual = x_visual.permute(1, 0, 2)  # NLD -> LND
        
        # text pre pro
        x_text = prompts_text + self.positional_embedding.type(self.dtype)
        x_text = x_text.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x_visual, x_text, compound_prompts_depth, 0]  # third argument is the counter which denotes depth of prompt
        
        
        outputs = self.resblocks(combined)
        
        # visual post_pro
        x_visual = outputs[0]
        x_visual = x_visual.permute(1, 0, 2)  # LND -> NLD

        x_visual = self.ln_post_visual(x_visual[:, 0, :])

        if self.proj_visual is not None:
            x_visual = x_visual @ self.proj_visual.half()

        # text post_pro
        x_text = outputs[1]  # extract the x back from here
        x_text = x_text.permute(1, 0, 2)  # LND -> NLD
        x_text = self.ln_final(x_text).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x_text = x_text[torch.arange(x_text.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x_visual, x_text
    
class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.CMPA.N_CTX
        ctx_init = cfg.TRAINER.CMPA.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0] # 512
        clip_imsize = clip_model.input_resolution # 224
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        # initialize text context
        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors_text = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors_text = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_text, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('CMPA design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of CMPA context words (tokens): {n_ctx}")
        self.prompts_text = nn.Parameter(ctx_vectors_text)
        
        # initialize visual context
        ctx_vectors_visual = torch.empty(n_ctx, 768, dtype=dtype)
        nn.init.normal_(ctx_vectors_visual, std=0.02)
        self.prompts_visual = nn.Parameter(ctx_vectors_visual)  

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.prompts_text

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts_text = self.construct_prompts(ctx, prefix, suffix)

        prompts_visual = self.prompts_visual
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompts_text, prompts_visual


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.CMPA.PROMPT_DEPTH >= 1, "For CMPA, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.CMPA.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.encoder = Encoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        compound_prompts_depth = self.compound_prompts_depth
        logit_scale = self.logit_scale.exp()

        prompts_text, prompts_visual = self.prompt_learner()
        image_features, text_features= self.encoder(image.type(self.dtype), prompts_visual, prompts_text, tokenized_prompts, compound_prompts_depth)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class CMPA(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.CMPA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CMPA.PREC == "fp32" or cfg.TRAINER.CMPA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                elif 'prompt_attn' in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.CMPA.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.CMPA.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
