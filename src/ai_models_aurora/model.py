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

    use_an = False

    def run(self):

        # TODO: control location of cache

        model = self.klass(use_lora=False)
        model = model.to(self.device)
        LOG.info("Downloading Aurora model %s", self.checkpoint)
        model.load_checkpoint("microsoft/aurora", self.checkpoint, strict=False)
        LOG.info("Loading Aurora model to device %s", self.device)

        model = model.to(self.device)
        model.eval()

        fields_pl = self.fields_pl
        fields_sfc = self.fields_sfc

        N, W, S, E = self.area
        WE, NS = self.grid
        Nj = round((N - S) / NS) + 1
        Ni = round((E - W) / WE) + 1

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

        LOG.info("Starting inference")
        with torch.inference_mode():

            with self.stepper(6) as stepper:
                for i, pred in enumerate(rollout(model, batch, steps=self.lead_time // 6)):
                    step = (i + 1) * 6

                    for k, v in pred.surf_vars.items():
                        v = v.cpu().numpy()
                        self.write(v, template=templates[k], step=step)

                    for k, v in pred.atmos_vars.items():
                        v = v.cpu().numpy()
                        for i, level in enumerate(self.levels):
                            self.write(v[:, :, i], template=templates[(k, level)], step=step)

                    stepper(i, step)


class Aurora2p5(AuroraModel):
    klass = Aurora
    download_files = ("aurora-0.25-static.pickle",)
    # Input
    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]


# https://microsoft.github.io/aurora/models.html#aurora-0-25-pretrained
class Aurora2p5Pretrained(Aurora2p5):
    checkpoint = "aurora-0.25-pretrained.ckpt"


class UseIFSMixin:
    def patch_retrieve_request(self, r):
        if r.get("class", "od") != "od":
            return

        if r.get("type", "an") not in ("an", "fc"):
            return

        if r.get("stream", "oper") not in ("oper", "scda"):
            return

        if self.use_an:
            r["type"] = "an"
        else:
            r["type"] = "fc"

        time = r.get("time", 12)

        r["stream"] = {
            0: "oper",
            6: "scda",
            12: "oper",
            18: "scda",
        }[time]


# https://microsoft.github.io/aurora/models.html#aurora-0-25-fine-tuned
class Aurora2p5FineTuned(UseIFSMixin, Aurora2p5):
    checkpoint = "aurora-0.25-finetuned.ckpt"


class Aurora0p1(AuroraModel):
    klass = AuroraHighRes

    download_files = ("aurora-0.1-static.pickle",)
    # Input
    area = [90, 0, -90, 360 - 0.1]
    grid = [0.1, 0.1]


# https://microsoft.github.io/aurora/models.html#aurora-0-1-fine-tuned
class Aurora0p1FineTuned(UseIFSMixin, Aurora0p1):
    klass = AuroraHighRes

    checkpoint = "aurora-0.1-finetuned.ckpt"


model = Aurora2p5FineTuned
