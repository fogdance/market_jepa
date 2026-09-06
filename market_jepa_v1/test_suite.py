"""Run every test with bounded DataLoader worker waits, without skipping assertions.

The execution sandbox can reject multiprocessing's local descriptor-sharing
socket. PyTorch's default timeout=0 then hangs forever instead of surfacing the
failure. This runner changes only that wait deadline, not workers, data or math.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from torch.utils.data import DataLoader


def main() -> int:
    original = DataLoader.__init__

    def bounded_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.num_workers > 0 and self.timeout == 0:
            self.timeout = 15

    with patch.object(DataLoader, "__init__", bounded_init):
        return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
