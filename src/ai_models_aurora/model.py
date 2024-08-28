# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

from anemoi.inference.plugin import AIModelPlugin
from aurora import Aurora, AuroraSmall, rollout

LOG = logging.getLogger(__name__)


class AuroraModel(AIModelPlugin):
    expver = "auro"


    # Download
    download_files = ["checkpoint.ckpt"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model =Aurora()

    def run(self):
        model = AuroraSmall()
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")

model = AuroraModel
