[bmprofile] is_mlir=1
...Start Profile Log...
[bmprofile] start to run subnet_id=0

[bmprofile] global_layer: layer_id=15 layer_type=MatMul layer_name=
[bmprofile] tensor_id=14 is_in=1 shape=[248320x2048] dtype=8 is_const=1 gaddr=4294967296 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=-1 is_in=1 shape=[1x2048] dtype=8 is_const=0 gaddr=5312086016 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=15 is_in=0 shape=[248320x1] dtype=8 is_const=0 gaddr=5312090112 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=2 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=3 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=1 gdma_id=2 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=1 gdma_id=4 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=1 gdma_id=5 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=2 gdma_id=3 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=2 gdma_id=6 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=2 gdma_id=7 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=3 gdma_id=5 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=3 gdma_id=8 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=3 gdma_id=9 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=4 gdma_id=7 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=4 gdma_id=10 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=4 gdma_id=11 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=5 gdma_id=9 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=5 gdma_id=12 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=5 gdma_id=13 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=6 gdma_id=11 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=6 gdma_id=14 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=6 gdma_id=15 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=7 gdma_id=13 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=7 gdma_id=16 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=7 gdma_id=17 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=8 gdma_id=15 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=8 gdma_id=18 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=8 gdma_id=19 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=9 gdma_id=17 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=9 gdma_id=20 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=9 gdma_id=21 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=10 gdma_id=19 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=10 gdma_id=22 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=10 gdma_id=23 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=11 gdma_id=21 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=11 gdma_id=24 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=11 gdma_id=25 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=12 gdma_id=23 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=12 gdma_id=26 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=12 gdma_id=27 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=13 gdma_id=25 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=13 gdma_id=28 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=13 gdma_id=29 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=14 gdma_id=27 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=14 gdma_id=30 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=14 gdma_id=31 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=15 gdma_id=29 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=15 gdma_id=32 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=15 gdma_id=33 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=16 gdma_id=31 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=16 gdma_id=34 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=16 gdma_id=35 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=17 gdma_id=33 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=17 gdma_id=36 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=17 gdma_id=37 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=18 gdma_id=35 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=18 gdma_id=38 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=18 gdma_id=39 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=19 gdma_id=37 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=19 gdma_id=40 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=19 gdma_id=41 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=20 gdma_id=39 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=20 gdma_id=42 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=20 gdma_id=43 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=21 gdma_id=41 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=21 gdma_id=44 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=21 gdma_id=45 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=22 gdma_id=43 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=22 gdma_id=46 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=22 gdma_id=47 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=23 gdma_id=45 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=23 gdma_id=48 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=23 gdma_id=49 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=24 gdma_id=47 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=24 gdma_id=50 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=24 gdma_id=51 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=25 gdma_id=49 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=25 gdma_id=52 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=25 gdma_id=53 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=26 gdma_id=51 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=26 gdma_id=54 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=26 gdma_id=55 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=27 gdma_id=53 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=27 gdma_id=56 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=27 gdma_id=57 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=28 gdma_id=55 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=28 gdma_id=58 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=28 gdma_id=59 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=29 gdma_id=57 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=29 gdma_id=60 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=29 gdma_id=61 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=30 gdma_id=59 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=30 gdma_id=62 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=30 gdma_id=63 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=31 gdma_id=61 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=31 gdma_id=64 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=31 gdma_id=65 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=32 gdma_id=63 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=32 gdma_id=66 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=32 gdma_id=67 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=33 gdma_id=65 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=33 gdma_id=68 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=33 gdma_id=69 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=34 gdma_id=67 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=34 gdma_id=70 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=34 gdma_id=71 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=35 gdma_id=69 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=35 gdma_id=72 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=35 gdma_id=73 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=36 gdma_id=71 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=36 gdma_id=74 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=36 gdma_id=75 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=37 gdma_id=73 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=37 gdma_id=76 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=37 gdma_id=77 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=38 gdma_id=75 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=38 gdma_id=78 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=38 gdma_id=79 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=39 gdma_id=77 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=39 gdma_id=80 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=39 gdma_id=81 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=40 gdma_id=79 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=40 gdma_id=82 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=40 gdma_id=83 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=41 gdma_id=81 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=41 gdma_id=84 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=41 gdma_id=85 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=42 gdma_id=83 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=42 gdma_id=86 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=42 gdma_id=87 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=43 gdma_id=85 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=43 gdma_id=88 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=43 gdma_id=89 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=44 gdma_id=87 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=44 gdma_id=90 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=44 gdma_id=91 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=45 gdma_id=89 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=45 gdma_id=92 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=45 gdma_id=93 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=46 gdma_id=91 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=46 gdma_id=94 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=46 gdma_id=95 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=47 gdma_id=93 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=47 gdma_id=96 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=47 gdma_id=97 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=48 gdma_id=95 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=48 gdma_id=98 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=48 gdma_id=99 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=49 gdma_id=97 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=49 gdma_id=100 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=49 gdma_id=101 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=50 gdma_id=99 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=50 gdma_id=102 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=50 gdma_id=103 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=51 gdma_id=101 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=51 gdma_id=104 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=51 gdma_id=105 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=52 gdma_id=103 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=52 gdma_id=106 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=52 gdma_id=107 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=53 gdma_id=105 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=53 gdma_id=108 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=53 gdma_id=109 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=54 gdma_id=107 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=54 gdma_id=110 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=111 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=55 gdma_id=109 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=55 gdma_id=112 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=55 gdma_id=113 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=56 gdma_id=111 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=56 gdma_id=114 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=56 gdma_id=115 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=57 gdma_id=113 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=57 gdma_id=116 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=57 gdma_id=117 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=58 gdma_id=115 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=58 gdma_id=118 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=58 gdma_id=119 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=59 gdma_id=117 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=59 gdma_id=120 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=59 gdma_id=121 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=60 gdma_id=119 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=60 gdma_id=122 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=60 gdma_id=123 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=61 gdma_id=121 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=61 gdma_id=124 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=125 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=62 gdma_id=123 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=62 gdma_id=126 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=62 gdma_id=127 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=63 gdma_id=125 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=63 gdma_id=128 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=63 gdma_id=129 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=64 gdma_id=127 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=64 gdma_id=130 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=64 gdma_id=131 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=65 gdma_id=129 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=65 gdma_id=132 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=65 gdma_id=133 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=66 gdma_id=131 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=66 gdma_id=134 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=135 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=67 gdma_id=133 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=67 gdma_id=136 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=67 gdma_id=137 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=68 gdma_id=135 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=68 gdma_id=138 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=68 gdma_id=139 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=69 gdma_id=137 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=69 gdma_id=140 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=69 gdma_id=141 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=70 gdma_id=139 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=70 gdma_id=142 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=70 gdma_id=143 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=71 gdma_id=141 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=71 gdma_id=144 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=71 gdma_id=145 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=72 gdma_id=143 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=72 gdma_id=146 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=72 gdma_id=147 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=73 gdma_id=145 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=73 gdma_id=148 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=73 gdma_id=149 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=74 gdma_id=147 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=74 gdma_id=150 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=74 gdma_id=151 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=75 gdma_id=149 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=75 gdma_id=152 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=75 gdma_id=153 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=76 gdma_id=151 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=76 gdma_id=154 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=76 gdma_id=155 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=77 gdma_id=153 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=77 gdma_id=156 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=77 gdma_id=157 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=78 gdma_id=155 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=78 gdma_id=158 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=78 gdma_id=159 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=79 gdma_id=157 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=79 gdma_id=160 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=79 gdma_id=161 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=80 gdma_id=159 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=80 gdma_id=162 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=80 gdma_id=163 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=81 gdma_id=161 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=81 gdma_id=164 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=81 gdma_id=165 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=82 gdma_id=163 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=82 gdma_id=166 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=82 gdma_id=167 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=83 gdma_id=165 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=83 gdma_id=168 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=83 gdma_id=169 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=84 gdma_id=167 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=84 gdma_id=170 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=84 gdma_id=171 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=85 gdma_id=169 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=85 gdma_id=172 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=85 gdma_id=173 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=86 gdma_id=171 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=86 gdma_id=174 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=175 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=87 gdma_id=173 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=87 gdma_id=176 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=87 gdma_id=177 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=88 gdma_id=175 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=88 gdma_id=178 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=88 gdma_id=179 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=89 gdma_id=177 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=89 gdma_id=180 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=89 gdma_id=181 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=90 gdma_id=179 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=90 gdma_id=182 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=90 gdma_id=183 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=91 gdma_id=181 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=91 gdma_id=184 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=91 gdma_id=185 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=92 gdma_id=183 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=92 gdma_id=186 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=92 gdma_id=187 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=93 gdma_id=185 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=93 gdma_id=188 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=93 gdma_id=189 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=94 gdma_id=187 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=94 gdma_id=190 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=94 gdma_id=191 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=95 gdma_id=189 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=95 gdma_id=192 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=95 gdma_id=193 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=96 gdma_id=191 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=96 gdma_id=194 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=96 gdma_id=195 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=97 gdma_id=193 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=97 gdma_id=196 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=97 gdma_id=197 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=98 gdma_id=195 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=98 gdma_id=198 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=98 gdma_id=199 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=99 gdma_id=197 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=99 gdma_id=200 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=99 gdma_id=201 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=100 gdma_id=199 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=100 gdma_id=202 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=100 gdma_id=203 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=101 gdma_id=201 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=101 gdma_id=204 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=101 gdma_id=205 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=102 gdma_id=203 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=102 gdma_id=206 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=102 gdma_id=207 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=103 gdma_id=205 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=103 gdma_id=208 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=103 gdma_id=209 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=104 gdma_id=207 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=104 gdma_id=210 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=104 gdma_id=211 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=105 gdma_id=209 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=105 gdma_id=212 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=105 gdma_id=213 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=106 gdma_id=211 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=106 gdma_id=214 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=106 gdma_id=215 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=107 gdma_id=213 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=107 gdma_id=216 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=107 gdma_id=217 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=108 gdma_id=215 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=108 gdma_id=218 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=108 gdma_id=219 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=109 gdma_id=217 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=109 gdma_id=220 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=109 gdma_id=221 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=110 gdma_id=219 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=110 gdma_id=222 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=110 gdma_id=223 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=111 gdma_id=221 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=111 gdma_id=224 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=111 gdma_id=225 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=112 gdma_id=223 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=112 gdma_id=226 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=112 gdma_id=227 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=113 gdma_id=225 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=113 gdma_id=228 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=113 gdma_id=229 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=114 gdma_id=227 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=114 gdma_id=230 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=114 gdma_id=231 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=115 gdma_id=229 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=115 gdma_id=232 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=115 gdma_id=233 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=116 gdma_id=231 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=116 gdma_id=234 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=116 gdma_id=235 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=117 gdma_id=233 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=117 gdma_id=236 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=117 gdma_id=237 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=118 gdma_id=235 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=118 gdma_id=238 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=118 gdma_id=239 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=119 gdma_id=237 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=119 gdma_id=240 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=119 gdma_id=241 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=120 gdma_id=239 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=120 gdma_id=242 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=120 gdma_id=243 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=121 gdma_id=241 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=121 gdma_id=244 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=121 gdma_id=245 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=122 gdma_id=243 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=122 gdma_id=246 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=122 gdma_id=247 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=123 gdma_id=245 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=123 gdma_id=248 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=123 gdma_id=249 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=124 gdma_id=247 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=124 gdma_id=250 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=124 gdma_id=251 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=125 gdma_id=249 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=125 gdma_id=252 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=125 gdma_id=253 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=126 gdma_id=251 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=126 gdma_id=254 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=126 gdma_id=255 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=127 gdma_id=253 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=127 gdma_id=256 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=127 gdma_id=257 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=128 gdma_id=255 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=128 gdma_id=258 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=128 gdma_id=259 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=129 gdma_id=257 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=129 gdma_id=260 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=129 gdma_id=261 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=130 gdma_id=259 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=130 gdma_id=262 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=130 gdma_id=263 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=131 gdma_id=261 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=131 gdma_id=264 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=131 gdma_id=265 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=132 gdma_id=263 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=132 gdma_id=266 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=132 gdma_id=267 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=133 gdma_id=265 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=133 gdma_id=268 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=133 gdma_id=269 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=134 gdma_id=267 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=134 gdma_id=270 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=134 gdma_id=271 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=135 gdma_id=269 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=135 gdma_id=272 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=135 gdma_id=273 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=136 gdma_id=271 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=136 gdma_id=274 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=136 gdma_id=275 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=137 gdma_id=273 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=137 gdma_id=276 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=137 gdma_id=277 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=138 gdma_id=275 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=138 gdma_id=278 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=138 gdma_id=279 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=139 gdma_id=277 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=139 gdma_id=280 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=139 gdma_id=281 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=140 gdma_id=279 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=140 gdma_id=282 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=140 gdma_id=283 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=141 gdma_id=281 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=141 gdma_id=284 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=141 gdma_id=285 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=142 gdma_id=283 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=142 gdma_id=286 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=142 gdma_id=287 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=143 gdma_id=285 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=143 gdma_id=288 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=144 gdma_id=287 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=144 gdma_id=289 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=16 layer_type=Reshape layer_name=
[bmprofile] tensor_id=15 is_in=1 shape=[248320x1] dtype=8 is_const=0 gaddr=5312090112 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=16 is_in=0 shape=[1x248320] dtype=8 is_const=0 gaddr=5312090112 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=17 layer_type=Cast layer_name=
[bmprofile] tensor_id=16 is_in=1 shape=[1x248320] dtype=8 is_const=0 gaddr=5312090112 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=17 is_in=0 shape=[1x248320] dtype=0 is_const=0 gaddr=5313085440 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=144 gdma_id=290 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=144 gdma_id=291 gdma_dir=0 gdma_func=1
[bmprofile] bd cmd_id bd_id=145 gdma_id=290 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=145 gdma_id=292 gdma_dir=0 gdma_func=1
[bmprofile] gdma cmd_id bd_id=145 gdma_id=293 gdma_dir=0 gdma_func=1
[bmprofile] bd cmd_id bd_id=146 gdma_id=291 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=146 gdma_id=294 gdma_dir=0 gdma_func=1
[bmprofile] bd cmd_id bd_id=147 gdma_id=293 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=147 gdma_id=295 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] insert bd end (cmd_id bd_id=148)
[bmprofile] bd cmd_id bd_id=148 gdma_id=0 bd_func=15
[bmprofile] insert gdma end (cmd_id gdma_id=296)
[bmprofile] gdma cmd_id bd_id=0 gdma_id=296 gdma_dir=0 gdma_func=6
[bmprofile] end to run subnet_id=0
