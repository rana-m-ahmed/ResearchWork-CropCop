from __future__ import annotations

"""Track-A v1.2 v8 operator authority.

This module deliberately leaves historical v1/v2/v3 operator files immutable while
rebinding their process-local science authority to the fully requalified v8 source.
Every v8 entry point imports this module before importing legacy helper modules.
"""

import tracka_v12_kaggle_operator as _v1
import tracka_v12_kaggle_operator_v2 as _v2
import tracka_v12_kaggle_operator_v3 as _v3

SCIENCE_SHA_V8 = "05ac7084a6be2fecd9c370477340ee0c8c4769bc"
OPERATOR_SCHEMA_VERSION_V8 = "4.0"

_v1.SCIENCE_SHA = SCIENCE_SHA_V8
_v2.SCIENCE_SHA = SCIENCE_SHA_V8
_v3.SCIENCE_SHA = SCIENCE_SHA_V8

from tracka_v12_kaggle_operator_v3 import *  # noqa: F401,F403,E402

SCIENCE_SHA = SCIENCE_SHA_V8
OPERATOR_SCHEMA_VERSION = OPERATOR_SCHEMA_VERSION_V8

if _v1.SCIENCE_SHA != SCIENCE_SHA or _v2.SCIENCE_SHA != SCIENCE_SHA or _v3.SCIENCE_SHA != SCIENCE_SHA:
    raise RuntimeError("v8 process-local science authority rebinding failed")
