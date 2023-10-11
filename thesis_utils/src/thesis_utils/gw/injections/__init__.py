"""Standard injections"""
import os
from pkg_resources import resource_filename

from .base import Injection

BBH_GW150914 = Injection(
    resource_filename(
        "thesis_utils", os.path.join("gw", "injections", "bbh_gw150914.json")
    )
)
BNS_VANILLA = Injection(
    resource_filename(
        "thesis_utils", os.path.join("gw", "injections", "bns_vanilla.json")
    )
)
