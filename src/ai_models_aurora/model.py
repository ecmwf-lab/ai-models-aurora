# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import pickle

import numpy as np
import torch
from ai_models.model import Model
from aurora import Batch
from aurora import Metadata
from aurora import rollout
from aurora.model.aurora import Aurora
from aurora.model.aurora import AuroraHighRes

LOG = logging.getLogger(__name__)


class AuroraModel(Model):

    download_url = "https://huggingface.co/microsoft/aurora/resolve/main/{file}"

    # Input
    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]

    surf_vars = ("2t", "10u", "10v", "msl")
    atmos_vars = ("z", "u", "v", "t", "q")
    levels = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)

    lagged = (-6, 0)

    #  For the MARS requets
    param_sfc = surf_vars
    param_level_pl = (atmos_vars, levels)

    # Output

    expver = "auro"
    lora = None

    def run(self):

        # TODO: control location of cache

        use_lora = self.lora if self.lora is not None else self.use_lora

        LOG.info(f"Model is {self.__class__.__name__}, use_lora={use_lora}")

        model = self.klass(use_lora=use_lora)
        model = model.to(self.device)

        path = os.path.join(self.assets, os.path.basename(self.checkpoint))
        if os.path.exists(path):
            LOG.info("Loading Aurora model from %s", path)
            model.load_checkpoint_local(path, strict=False)
        else:
            LOG.info("Downloading Aurora model %s", self.checkpoint)
            try:
                model.load_checkpoint("microsoft/aurora", self.checkpoint, strict=False)
            except Exception:
                LOG.error("Did not find a local copy at %s", path)
                raise

        LOG.info("Loading Aurora model to device %s", self.device)

        model = model.to(self.device)
        model.eval()

        fields_pl = self.fields_pl
        fields_sfc = self.fields_sfc

        Nj, Ni = fields_pl[0].shape

        to_numpy_kwargs = dict(dtype=np.float32)

        templates = {}

        # Shape (Batch, Time, Lat, Lon)
        surf_vars = {}

        for k in self.surf_vars:
            f = fields_sfc.sel(param=k).order_by(valid_datetime="ascending")
            templates[k] = f[-1]
            f = f.to_numpy(**to_numpy_kwargs)
            f = torch.from_numpy(f)
            f = f.unsqueeze(0)  # Add batch dimension
            surf_vars[k] = f

        # Shape (Lat, Lon)
        static_vars = {}
        with open(os.path.join(self.assets, self.download_files[0]), "rb") as f:
            static_vars = pickle.load(f)
            for k, v in static_vars.items():
                static_vars[k] = torch.from_numpy(v)

        # Shape (Batch, Time, Level, Lat, Lon)
        atmos_vars = {}
        for k in self.atmos_vars:
            f = fields_pl.sel(param=k).order_by(valid_datetime="ascending", level=self.levels)

            for level in self.levels:
                templates[(k, level)] = f.sel(level=level)[-1]

            f = f.to_numpy(**to_numpy_kwargs).reshape(len(self.lagged), len(self.levels), Nj, Ni)
            f = torch.from_numpy(f)
            f = f.unsqueeze(0)  # Add batch dimension
            atmos_vars[k] = f

        self.write_input_fields(fields_pl + fields_sfc)

        # https://microsoft.github.io/aurora/batch.html

        N, W, S, E = self.area

        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=torch.linspace(N, S, Nj),
                lon=torch.linspace(W, E, Ni),
                time=(self.start_datetime,),
                atmos_levels=self.levels,
            ),
        )

        assert len(batch.metadata.lat) == Nj
        assert len(batch.metadata.lon) == Ni

        LOG.info("Starting inference")
        with torch.inference_mode():

            with self.stepper(6) as stepper:
                for i, pred in enumerate(rollout(model, batch, steps=self.lead_time // 6)):
                    step = (i + 1) * 6

                    for k, v in pred.surf_vars.items():
                        data = np.squeeze(v.cpu().numpy())
                        data = self.nan_extend(data)
                        assert data.shape == (Nj, Ni)
                        self.write(data, template=templates[k], step=step, check_nans=True)

                    for k, v in pred.atmos_vars.items():
                        v = v.cpu().numpy()
                        for j, level in enumerate(self.levels):
                            data = np.squeeze(v[:, :, j])
                            data = self.nan_extend(data)
                            assert data.shape == (Nj, Ni)
                            self.write(data, template=templates[(k, level)], step=step, check_nans=True)
                    stepper(i, step)

    def nan_extend(self, data):
        return np.concatenate(
            (data, np.full_like(data[[-1], :], np.nan, dtype=data.dtype)),
            axis=0,
        )

    def parse_model_args(self, args):
        import argparse

        parser = argparse.ArgumentParser("ai-models aurora")

        parser.add_argument(
            "--lora",
            type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
            nargs="?",
            const=True,
            default=None,
            help="Use LoRA model (true/false). Default depends on the model.",
        )

        return parser.parse_args(args)


class Aurora0p25(AuroraModel):
    klass = Aurora
    download_files = ("aurora-0.25-static.pickle",)
    # Input
    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]


# https://microsoft.github.io/aurora/models.html#aurora-0-25-pretrained
class Aurora0p25Pretrained(Aurora0p25):
    use_lora = False
    checkpoint = "aurora-0.25-pretrained.ckpt"


# https://microsoft.github.io/aurora/models.html#aurora-0-25-fine-tuned
class Aurora025FineTuned(Aurora0p25):
    use_lora = True
    checkpoint = "aurora-0.25-finetuned.ckpt"

    # We want FC, step=0
    def patch_retrieve_request(self, r):
        if r.get("class", "od") != "od":
            return

        if r.get("type", "an") not in ("an", "fc"):
            return

        if r.get("stream", "oper") not in ("oper", "scda"):
            return

        r["type"] = "fc"

        time = r.get("time", 12)

        r["stream"] = {
            0: "oper",
            6: "scda",
            12: "oper",
            18: "scda",
        }[time]


# https://microsoft.github.io/aurora/models.html#aurora-0-1-fine-tuned
class Aurora0p1FineTuned(AuroraModel):
    download_files = ("aurora-0.1-static.pickle",)
    # Input
    area = [90, 0, -90, 360 - 0.1]
    grid = [0.1, 0.1]

    klass = AuroraHighRes
    use_lora = True
    checkpoint = "aurora-0.1-finetuned.ckpt"


# model = Aurora0p1FineTuned


def model(model_version, **kwargs):

    # select with --model-version

    models = {
        "0.25-pretrained": Aurora0p25Pretrained,
        "0.25-finetuned": Aurora025FineTuned,
        "0.1-finetuned": Aurora0p1FineTuned,
        "default": Aurora0p1FineTuned,
        "latest": Aurora0p1FineTuned,  # Backward compatibility
    }

    if model_version not in models:
        LOG.error(f"Model version {model_version} not found, using default")
        LOG.error(f"Available models: {list(models.keys())}")
        raise ValueError(f"Model version {model_version} not found")

    return models[model_version](**kwargs)
