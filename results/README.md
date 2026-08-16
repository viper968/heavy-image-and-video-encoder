# Benchmark output

Raw output from the benchmark tools, regenerated with:

    python tools/fetch_testdata.py
    python tools/ladder.py       testdata/images/*.png          > results/ladder.txt
    python tools/bench_image.py --jobs=12 dev                   > results/image_kodak.txt
    python tools/bench_image.py --jobs=12 test                  > results/image_heldout.txt
    python tools/bench_video.py  testdata/video/akiyo_cif.y4m   16 > results/video_akiyo.txt
    python tools/bench_video.py  testdata/video/foreman_cif.y4m 16 > results/video_foreman.txt

`image_kodak.txt` is the six-image **dev** split and `image_heldout.txt` is the
eighteen-image **held-out** split; the README's headline number is the held-out
one. They are easy to confuse and have been swapped by accident before, so check
the `split:` line at the top of the file before trusting a number from it.

`video_sintel_1080p.txt` uses the first 16 frames of
`sintel_trailer_2k_1080p24.y4m` from media.xiph.org, which `fetch_testdata.py`
does not download — it is several GB. Fetch just the head of it:

    curl -r 0-73400320 https://media.xiph.org/video/derf/y4m/sintel_trailer_2k_1080p24.y4m \
        -o sintel.y4m
    python tools/bench_video.py sintel.y4m 16 > results/video_sintel_1080p.txt

Every codec in these tables was decoded and compared against the source before
its size was recorded. The `cpu s` / `enc s` columns move by 10-20% between runs
on the same code; the byte counts do not.
