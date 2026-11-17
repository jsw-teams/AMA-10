#ifndef AMA10_SWANLAB_H
#define AMA10_SWANLAB_H

#include <stddef.h>

/* Upload scalar metrics through SwanLab's HTTP API (no Python SDK required).
   The API key is read from SWANLAB_API_KEY; base_url is the server origin,
   for example https://api.swanlab.cn. A project must already exist. */
int ama10_swanlab_upload_csv(const char *base_url, const char *workspace,
                             const char *project, const char *experiment,
                             const char *csv_path, char *error,
                             size_t error_capacity);

#endif
