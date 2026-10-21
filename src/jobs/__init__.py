"""Long-running work as jobs: the batch lane beside the interactive request path.

A transcription answers in seconds and runs inside the HTTP request. Speaker
separation of a whole recording takes minutes and produces gigabytes of
tracks, so it runs as a job: ``POST /v1/audio/transcriptions`` with
``separate=true`` answers ``202`` with a job id, the batch lane runs the job on
its own worker thread with batch priority on the GPU tokens, and the client
polls ``GET /v1/jobs/{id}`` and downloads the tracks when it is done.

- :mod:`.store`  — job rows in SQLite under ``JOBS_DIR`` plus one directory per job
- :mod:`.lane`   — the bounded queue and worker threads
- :mod:`.separate_job` — what a separation job does, stage by stage
- :mod:`.service` — ties the three together for the HTTP layer
"""
