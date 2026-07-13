#!/bin/sh
# gdb wrapper for the VS Code cppdbg sessions: cpptools force-enables
# debuginfod (MI command "set debuginfod enabled on"), and with Ubuntu's
# system-wide DEBUGINFOD_URLS that makes symbol loading query the network
# for the Release binary and every solib -- the debug session then hangs
# for minutes at -file-exec-and-symbols with no window and no error.
# Unsetting the URLs makes the forced enable a no-op. (Plain terminal gdb
# never hits this: interactive/batch gdb defaults debuginfod OFF.)
unset DEBUGINFOD_URLS
exec gdb "$@"
