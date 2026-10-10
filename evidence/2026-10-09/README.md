# Evidence for image 2026-10-09 (four DGX Sparks, CONTEXT=163840, 2026-10-09)

- `repro-checks.txt` / `repro-client.log`: thinking-off prompt tail, streamed and non-streamed client disconnect,
  and the field repro (a title request cut by a foreground request, its client leaving, a second request arriving). All PASS.
- `one-shot-bench.txt`: `./one-shot.sh bench` on the same boot.
- `watchdog-pause-rank2.log`: `TF_GLM_MULTI_WATCHDOG_S=60`, rank 2 frozen with `docker pause` during a request;
  ranks 0, 1 and 3 exited 68 s later. `watchdog-rank*-tail.log`: the stack dumps they wrote.
- `one-shot-watch.log`: `./one-shot.sh watch` saw the exits, restarted all four and the server answered again.
(the last 'down' and 'giving up' lines are the test's own teardown: the test stops the server with MAX_RESTARTS=1.)
