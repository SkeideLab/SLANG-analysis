#!/bin/bash

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# Clean up the temporary directory
tmp_dir="/ptmp/$USER/tmp/"
chmod -R +w "$tmp_dir"
rm -rf "$tmp_dir"
