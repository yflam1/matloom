from os import makedirs
from os.path import dirname, isfile
from typing import Annotated, Any, Literal

import torch
from pydantic import Field, validate_call
from typing_extensions import Self

from ..utils.anybase import AnyBase
from ..utils.dtypes import ImgLike, SDict
from ..utils.misc import finalize_device
from .base import VlmResult


class ClipScore(AnyBase):
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        model: torch.nn.Module,
        preprocess,
        tokenizer,
        *,
        label: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._device = next(model.parameters()).device
        self._label = self.cname_ if label is None else label

    @torch.inference_mode()
    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __call__(
        self,
        images: list[ImgLike],
        texts: list[str],
        *,
        w: Annotated[float, Field(gt=0.0)] = 100.0,
        progress: bool = True,
    ) -> VlmResult:
        from ..utils.file import load_image_to_pillow
        from ..utils.progress import progress_bar

        scores = torch.empty((len(texts), len(images)))
        with progress_bar(disable=not progress) as pbar:
            # Initialize progress bar
            task_id = pbar.add_task(f"{self._label}", total=len(images))

            # Preprocess texts
            txt_token = self._tokenizer(texts).to(self.device)
            txt_emb: torch.Tensor = self._model.encode_text(txt_token)
            txt_emb = torch.nn.functional.normalize(txt_emb.detach())

            for i, img in enumerate(images):
                img_tensor: torch.Tensor = (
                    self._preprocess(load_image_to_pillow(img))
                    .unsqueeze(0)
                    .to(self.device)
                )
                img_emb: torch.Tensor = self._model.encode_image(img_tensor)
                img_emb = torch.nn.functional.normalize(img_emb.detach())
                score = txt_emb @ img_emb.T
                scores[:, i] = score.squeeze(-1).clamp(0.0, 1.0).cpu() * w

                # Update progress bar
                pbar.advance(task_id)

            # Remove progress bar task
            pbar.remove_task(task_id)
        return VlmResult(scores=scores)

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    @validate_call
    def _from_OpenClip(
        cls,
        name: str,
        pretrained: str,
        device: str | None = None,
        cache_dir: str | None = None,
        label: str | None = None,
    ) -> Self:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            name,
            pretrained=pretrained,
            device=finalize_device(device, mps_ok=True),
            cache_dir=cache_dir,
        )
        model.eval()
        tokenizer = open_clip.get_tokenizer(name, cache_dir=cache_dir)
        return cls(model, preprocess, tokenizer, label=label)

    @classmethod
    def ClipA(
        cls,
        name: str = "ViT-L-14-CLIPA",  # 224
        pretrained: str = "datacomp1b",
        device: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> Self:
        return cls._from_OpenClip(
            name,
            pretrained,
            device=device,
            cache_dir=cache_dir,
            label=cls.get_method_name(),
        )

    @classmethod
    def Dfn(
        cls,
        name: str = "ViT-L-14-quickgelu",  # 224
        pretrained: str = "dfn2b",
        device: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> Self:
        return cls._from_OpenClip(
            name,
            pretrained,
            device=device,
            cache_dir=cache_dir,
            label=cls.get_method_name(),
        )

    @classmethod
    def EvaClip(
        cls,
        name: str = "EVA02-L-14",  # 224
        pretrained: str = "merged2b_s4b_b131k",
        device: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> Self:
        return cls._from_OpenClip(
            name,
            pretrained,
            device=device,
            cache_dir=cache_dir,
            label=cls.get_method_name(),
        )

    @classmethod
    @validate_call
    def LongClip(
        cls,
        name: Literal["longclip-B", "longclip-L"] = "longclip-L",  # 224
        device: str | None = None,
        **kwargs: Any,
    ) -> Self:
        import urllib.request

        from ..models.longclip import longclip
        from ..utils.misc import loading

        model_urls = {
            "longclip-B": "https://huggingface.co/BeichenZhang/LongCLIP-B/resolve/main/longclip-B.pt",
            "longclip-L": "https://huggingface.co/BeichenZhang/LongCLIP-L/resolve/main/longclip-L.pt",
        }

        # Check if the model exists in the current directory
        path = f"{name}.pt"
        if not isfile(path):
            url = model_urls[name]
            with loading(f"Downloading {url}..."):
                urllib.request.urlretrieve(url, path)

        model, preprocess = longclip.load(
            path, device=finalize_device(device, mps_ok=True)
        )
        return cls(
            model,
            preprocess,
            lambda x: longclip.tokenize(x, truncate=True),
            label=cls.get_method_name(),
        )

    @classmethod
    def MetaClip(
        cls,
        name: str = "ViT-L-14-quickgelu",  # 224
        pretrained: str = "metaclip_400m",
        device: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> Self:
        return cls._from_OpenClip(
            name,
            pretrained,
            device=device,
            cache_dir=cache_dir,
            label=cls.get_method_name(),
        )

    @classmethod
    @validate_call
    def MobileClip2(
        cls,
        variant: Literal["B", "S0", "S2", "S3", "S4", "L-14"] = "S2",  # 256
        device: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> Self:
        return cls._from_OpenClip(
            f"MobileCLIP2-{variant}",
            "dfndr2b",
            device=device,
            cache_dir=cache_dir,
            label=cls.get_method_name(),
        )

    @classmethod
    @validate_call
    def OpenAi(
        cls,
        name: str = "ViT-L/14",  # 224
        device: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> Self:
        import clip

        model, preprocess = clip.load(
            name, device=finalize_device(device, mps_ok=True), download_root=cache_dir
        )
        return cls(
            model,
            preprocess,
            lambda x: clip.tokenize(x, truncate=True),
            label=cls.get_method_name(),
        )

    @classmethod
    def OpenClip(
        cls,
        name: str = "ViT-L-14",  # 224
        pretrained: str = "laion2b_s32b_b82k",
        device: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> Self:
        return cls._from_OpenClip(
            name,
            pretrained,
            device=device,
            cache_dir=cache_dir,
            label=cls.get_method_name(),
        )

    @classmethod
    def SigLip(
        cls,
        name: str = "ViT-L-16-SigLIP-256",  # 256
        pretrained: str = "webli",
        device: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> Self:
        return cls._from_OpenClip(
            name,
            pretrained,
            device=device,
            cache_dir=cache_dir,
            label=cls.get_method_name(),
        )

    @classmethod
    def SigLip2(
        cls,
        name: str = "ViT-L-16-SigLIP2-256",  # 256
        pretrained: str = "webli",
        device: str | None = None,
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> Self:
        return cls._from_OpenClip(
            name,
            pretrained,
            device=device,
            cache_dir=cache_dir,
            label=cls.get_method_name(),
        )


def _cli() -> None:
    import argparse
    import ast
    import csv
    from glob import glob
    from itertools import product

    from ..utils.file import read_file
    from ..utils.misc import get_datetime, get_device

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", action="append", type=str, required=True)
    parser.add_argument("-pt", "--prompt_text", action="append", type=str, default=[])
    parser.add_argument("-pf", "--prompt_file", action="append", type=str, default=[])
    parser.add_argument("-m", "--method", type=str, default="OpenClip")
    parser.add_argument("--kwargs", type=str, default=r"{}")
    parser.add_argument(
        "-o",
        "--output_csv",
        type=str,
        default=f"clipscore_{get_datetime()}.csv",
    )
    args = parser.parse_args()
    if len(args.prompt_text) == 0 and len(args.prompt_file) == 0:
        raise ValueError(
            "At least one of `prompt_text` or `prompt_file` must be provided"
        )
    kwargs: SDict[Any] = ast.literal_eval(args.kwargs)
    clip: ClipScore = getattr(ClipScore, args.method)(**kwargs, device=get_device()[0])
    texts: list[str] = args.prompt_text
    for pf in args.prompt_file:
        paths = sorted(glob(pf))
        for path in paths:
            texts.extend(read_file(path, remove_spaces=True, remove_empty=True))
    output_dir = dirname(args.output_csv)
    if output_dir not in {"", "."}:
        makedirs(output_dir, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "text", "image", "score"])
        for img in args.image:
            images = sorted(glob(img))
            result = clip(images, texts).scores
            for i, j in product(range(result.size(0)), range(result.size(1))):
                writer.writerow([args.method, texts[i], images[j], result[i, j].item()])


if __name__ == "__main__":
    _cli()
