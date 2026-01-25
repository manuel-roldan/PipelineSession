Neat plugins binary drop

This directory holds prebuilt GStreamer plugins:
  - `libgstneatdecoder.so`
  - `libgstneatencoder.so`

Stage binaries here after building.

Then load them for GStreamer:
  `source scripts/use_neatdecoder.sh`

Install both system-wide:
  `sudo scripts/install_neat_plugins.sh`

If you need to override the system plugin path:
  `SIMA_SET_GST_SYSTEM_PATH=1 source scripts/use_neatdecoder.sh`
