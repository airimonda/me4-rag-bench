# Troubleshooting

Find your message, follow the numbered steps. Run `crb check` after each fix.

## Adapter not reachable

`crb check` says it cannot reach your adapter.

1. Is your server running? Look in the terminal where you started it.
2. Test it by hand:
   ```bash
   curl http://127.0.0.1:8000/health
   ```
   You should see `{"ok": true, ...}`.
3. Different port? Tell `crb`: `crb run --adapter http://127.0.0.1:9000`.
4. Server listens on `localhost` only for IPv6? Start it on `127.0.0.1`.
5. Server written in Docker? Publish the port: `-p 8000:8000`.

## An endpoint fails the probe

`crb check` names the endpoint. Run the curl example for it in [ADAPTER.md](ADAPTER.md) and read your server's output. Common causes: the reply is not JSON, a missing `Content-Type`, or an exception in your code. `/retrieve` may return `{"pages": null}` if you have no retriever. That is allowed.

## Port in use

"Address already in use" on 8000, 8790 or 8302.

1. Find the program:
   ```bash
   ss -ltnp | grep ':8000'
   ```
2. Stop it (`kill <pid>`), or choose another port: `crb baseline` has `--port`, `crb serve --port 8303`.
3. Port 8790 belongs to the token proxy. A leftover `crb` is probably still running: `pkill -f "crb run"`.

## No vcgencmd

`vcgencmd` is part of Raspberry Pi OS. If it is missing the report shows no temperature or throttling.

1. Check: `which vcgencmd`.
2. Install: `sudo apt install libraspberrypi-bin`.
3. If the command exists but fails with a permission error, add yourself to the video group and log in again: `sudo usermod -aG video $USER`.

On a laptop it is normal. Health columns are then empty and the report says so.

## Under-voltage

The report shows "under-voltage" or `throttled_hex` has bit 0. The power supply cannot deliver enough current.

1. Use the official Pi power supply (5.1 V 3 A for Pi 4, 5.1 V 5 A for Pi 5).
2. Use a short, thick USB cable. Do not power the Pi from a laptop port.
3. Unplug extras (USB drives, cameras) to see if it stops.
4. Run again. Numbers from an under-volted run are lower than your system really is.

## Throttling and cooling

The report shows seconds throttled or a peak temperature near 80 °C. The Pi slows itself down when hot.

1. Fit a heatsink and a fan (a Pi 5 needs the active cooler).
2. Take the Pi out of a closed case. Keep it away from sun and other heat.
3. Check the temperature live: `watch -n 1 vcgencmd measure_temp`.
4. Let it cool to below 50 °C before a new run.
5. Check there is no leftover heavy process: `top`.

## Disk full

Needs about 2.5 GB free for the data, plus space for your own index.

1. See what is free: `df -h ~`.
2. Remove old runs you do not need: `rm -r runs/<student>/<old-run>`.
3. Clean the package cache: `sudo apt clean` and `pip cache purge`.
4. Move data to a USB disk: `export CRB_DATA=/mnt/usb/crb-data` then `crb fetch`.
5. Do not delete `~/.cache/crb/judge.sqlite`. It saves you money when you re-score.

## Judge key missing

`crb check` says no judge key found.

1. Set the key in the terminal you run `crb` from:
   ```bash
   export OPENROUTER_API_KEY=your-key-here
   ```
   or, if the teaching team gave you a separate key: `export CRB_JUDGE_KEY=...`.
2. `export` lasts only for this terminal. Put the line in `~/.bashrc` to keep it.
3. Without a key you can still run `crb run --no-judge`, then `crb score RUN_DIR` later.

## Resume a stopped run

Power cut, Ctrl-C, SSH dropped? Nothing is lost.

1. Find the run folder: `ls runs/<student>/`.
2. Restart your adapter if needed. Your index must be rebuilt after a restart; `crb` calls `/ingest` again.
3. Run:
   ```bash
   crb run --resume runs/12/20261127-101500
   ```
   Questions already in `answers.jsonl` are skipped.

Tip: use `tmux` (`sudo apt install tmux`, then `tmux`) so the run survives a dropped SSH connection.

## Cannot open the report from my laptop

1. On the Pi: `crb serve`. Open the link it prints.
2. Same network? A school or guest network may block device-to-device traffic. Use a phone hotspot, or Tailscale.
3. Or copy the file: `scp pi@raspberrypi.local:~/me4-rag-bench/runs/12/*/report.html .`

## Answers are slow or time out

1. Open the report. Is the Pi hot or under-volted? Fix that first.
2. Look at the network round trip in the health section. A slow link makes a fast model look slow.
3. Stream your answer. Time to first token is the number people feel.
4. `/answer` times out at 120 seconds and `/retrieve` at 30. A timeout is recorded as a failed answer and the run goes on.

## Still stuck

Send the teaching team: the output of `crb check`, and `run.log` from your run folder. No keys are in either file.
