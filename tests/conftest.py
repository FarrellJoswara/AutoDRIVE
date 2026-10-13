"""Keep imported Mission Control modules isolated from the user's live runtime."""

import os


# Several route tests import src.layer4.hub.app at collection time. The test
# backend prevents those imports from adopting native process manifests or
# reconciling the workspace's persistent run registry.
os.environ["AICAR_RUN_BACKEND"] = "test"
