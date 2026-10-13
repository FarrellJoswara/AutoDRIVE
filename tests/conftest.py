"""Keep imported Mission Control modules isolated from the user's live runtime."""

import os
import sys

from process_spawn_guard import reject_protected_runtime_spawn


# Several route tests import src.layer4.hub.app at collection time. The test
# backend prevents those imports from adopting native process manifests or
# reconciling the workspace's persistent run registry.
os.environ["AICAR_RUN_BACKEND"] = "test"

# The test backend only isolates Mission Control's app factory.  It does not
# protect direct calls to Racer or NativeOfficialBackend, so enforce the same
# boundary at Python's process-creation audit event.  Unlike monkeypatching
# subprocess.Popen, this remains active when a test patches an imported alias.
sys.addaudithook(reject_protected_runtime_spawn)
