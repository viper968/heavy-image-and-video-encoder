# Benchmark output

Raw output from the benchmark tools, regenerated with:

    python tools/fetch_testdata.py
    python tools/ladder.py       testdata/images/*.png       > results/ladder.txt
    python tools/bench_image.py  testdata/images/*.png       > results/image_kodak.txt
    python tools/bench_video.py  testdata/video/akiyo_cif.y4m   16 > results/video_akiyo.txt
    python tools/bench_video.py  testdata/video/foreman_cif.y4m 16 > results/video_foreman.txt

Every codec in these tables was decoded and compared against the source before
its size was recorded.
