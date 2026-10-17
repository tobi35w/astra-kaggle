"""Tie Windows verification children to the lifetime of their desk process.

A private, non-inheritable Windows job terminates attached children and their
descendants when the desk exits, including abrupt process termination. Other
platforms retain normal subprocess cleanup but have no abrupt-parent-exit guard.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys


class _Limits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64),
        ("job_time", ctypes.c_int64),
        ("flags", ctypes.c_uint32),
        ("minimum_working_set", ctypes.c_size_t),
        ("maximum_working_set", ctypes.c_size_t),
        ("active_processes", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority", ctypes.c_uint32),
        ("scheduling", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "read_operations",
            "write_operations",
            "other_operations",
            "read_bytes",
            "write_bytes",
            "other_bytes",
        )
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", _Limits),
        ("io", _IoCounters),
        ("process_memory", ctypes.c_size_t),
        ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


class ChildLifetime:
    """Create before Popen; attach immediately, before consuming any output."""

    def __init__(self):
        self._handle = None
        self._api = None
        if sys.platform != "win32":
            return
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        api.CreateJobObjectW.restype = ctypes.c_void_p
        api.SetInformationJobObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        api.SetInformationJobObject.restype = ctypes.c_int
        api.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        api.AssignProcessToJobObject.restype = ctypes.c_int
        api.CloseHandle.argtypes = [ctypes.c_void_p]
        api.CloseHandle.restype = ctypes.c_int
        handle = api.CreateJobObjectW(None, None)
        if not handle:
            raise OSError("Cannot create verification process guard")
        limits = _ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(
            handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            api.CloseHandle(handle)
            raise OSError("Cannot configure verification process guard")
        self._api = api
        self._handle = handle

    def attach(self, process: subprocess.Popen) -> None:
        if self._api is None:
            return
        # Use Popen's retained process handle, never reopen an integer PID that
        # may have been recycled after a very short command exits.
        handle = int(process._handle)  # type: ignore[attr-defined]  # CPython Windows handle.
        if (
            not self._api.AssignProcessToJobObject(self._handle, handle)
            and process.poll() is None
        ):
            raise OSError("Cannot attach verification process guard")

    def close(self) -> None:
        if self._api is not None and self._handle is not None:
            self._api.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
