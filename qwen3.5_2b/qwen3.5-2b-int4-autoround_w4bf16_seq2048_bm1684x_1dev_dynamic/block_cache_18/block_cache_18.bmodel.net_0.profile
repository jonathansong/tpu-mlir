[bmprofile] is_mlir=1
...Start Profile Log...
[bmprofile] start to run subnet_id=0

[bmprofile] global_layer: layer_id=47 layer_type=RMSNorm layer_name=
[bmprofile] tensor_id=-1 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4325937152 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=16 is_in=1 shape=[1x1x2048] dtype=8 is_const=1 gaddr=4294967296 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=47 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326526976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=0 gdma_id=1 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=0 gdma_id=2 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=1 gdma_id=2 bd_func=3
[bmprofile] bd cmd_id bd_id=2 gdma_id=2 bd_func=3
[bmprofile] bd cmd_id bd_id=3 gdma_id=2 bd_func=1
[bmprofile] bd cmd_id bd_id=4 gdma_id=2 bd_func=3
[bmprofile] bd cmd_id bd_id=5 gdma_id=2 bd_func=9
[bmprofile] bd cmd_id bd_id=6 gdma_id=2 bd_func=13
[bmprofile] bd cmd_id bd_id=7 gdma_id=2 bd_func=3
[bmprofile] bd cmd_id bd_id=8 gdma_id=2 bd_func=3
[bmprofile] bd cmd_id bd_id=9 gdma_id=2 bd_func=5
[bmprofile] bd cmd_id bd_id=10 gdma_id=2 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=10 gdma_id=3 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=48 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=47 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326526976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=28 is_in=1 shape=[2048x1024] dtype=3 is_const=1 gaddr=4301852672 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=27 is_in=1 shape=[64x32x16] dtype=8 is_const=1 gaddr=4301787136 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=26 is_in=1 shape=[64x32x16] dtype=3 is_const=1 gaddr=4301754368 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=48 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=10 gdma_id=4 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=10 gdma_id=5 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=10 gdma_id=6 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=10 gdma_id=7 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=10 gdma_id=8 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=11 gdma_id=7 bd_func=3
[bmprofile] bd cmd_id bd_id=12 gdma_id=7 bd_func=3
[bmprofile] bd cmd_id bd_id=13 gdma_id=7 bd_func=3
[bmprofile] bd cmd_id bd_id=14 gdma_id=7 bd_func=3
[bmprofile] bd cmd_id bd_id=15 gdma_id=7 bd_func=3
[bmprofile] bd cmd_id bd_id=16 gdma_id=7 bd_func=5
[bmprofile] bd cmd_id bd_id=17 gdma_id=7 bd_func=2
[bmprofile] bd cmd_id bd_id=18 gdma_id=7 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=18 gdma_id=9 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=19 gdma_id=8 bd_func=3
[bmprofile] bd cmd_id bd_id=20 gdma_id=8 bd_func=3
[bmprofile] bd cmd_id bd_id=21 gdma_id=8 bd_func=3
[bmprofile] bd cmd_id bd_id=22 gdma_id=8 bd_func=3
[bmprofile] bd cmd_id bd_id=23 gdma_id=8 bd_func=3
[bmprofile] bd cmd_id bd_id=24 gdma_id=8 bd_func=2
[bmprofile] bd cmd_id bd_id=25 gdma_id=8 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=25 gdma_id=10 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=49 layer_type=Cast layer_name=
[bmprofile] tensor_id=48 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=49 is_in=0 shape=[1x1x2048] dtype=0 is_const=0 gaddr=4326576128 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=25 gdma_id=11 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=26 gdma_id=11 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=26 gdma_id=12 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=51 layer_type=Load layer_name=
[bmprofile] tensor_id=20 is_in=1 shape=[1x1x16] dtype=8 is_const=1 gaddr=4295032832 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=51 is_in=0 shape=[1x1x16] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=26 gdma_id=13 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=52 layer_type=Load layer_name=
[bmprofile] tensor_id=21 is_in=1 shape=[1x2048x16] dtype=8 is_const=1 gaddr=4295036928 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=52 is_in=0 shape=[1x2048x16] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=26 gdma_id=14 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=53 layer_type=Load layer_name=
[bmprofile] tensor_id=47 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326526976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=53 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=26 gdma_id=15 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=55 layer_type=MatMul layer_name=
[bmprofile] tensor_id=53 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=52 is_in=1 shape=[1x2048x16] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=51 is_in=1 shape=[1x1x16] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=55 is_in=0 shape=[1x1x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=27 gdma_id=15 bd_func=5
[bmprofile] bd cmd_id bd_id=28 gdma_id=15 bd_func=2
[bmprofile] bd cmd_id bd_id=29 gdma_id=15 bd_func=3
[bmprofile] local_layer: layer_id=54 layer_type=Load layer_name=
[bmprofile] tensor_id=22 is_in=1 shape=[1x2048x16] dtype=8 is_const=1 gaddr=4295102464 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=54 is_in=0 shape=[1x2048x16] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=26 gdma_id=16 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=56 layer_type=Cast layer_name=
[bmprofile] tensor_id=55 is_in=1 shape=[1x1x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=56 is_in=0 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=34816 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=30 gdma_id=16 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=58 layer_type=MatMul layer_name=
[bmprofile] tensor_id=53 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=54 is_in=1 shape=[1x2048x16] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=58 is_in=0 shape=[1x1x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49216 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=31 gdma_id=16 bd_func=2
[bmprofile] local_layer: layer_id=57 layer_type=Store layer_name=
[bmprofile] tensor_id=56 is_in=1 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=34816 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=57 is_in=0 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=34816 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=30 gdma_id=17 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=59 layer_type=Cast layer_name=
[bmprofile] tensor_id=58 is_in=1 shape=[1x1x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49216 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=59 is_in=0 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=18432 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=32 gdma_id=17 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=60 layer_type=Store layer_name=
[bmprofile] tensor_id=59 is_in=1 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=18432 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=60 is_in=0 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=18432 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=32 gdma_id=18 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=63 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=47 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326526976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=25 is_in=1 shape=[6144x1024] dtype=3 is_const=1 gaddr=4295462912 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=24 is_in=1 shape=[64x96x16] dtype=8 is_const=1 gaddr=4295266304 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=23 is_in=1 shape=[64x96x16] dtype=3 is_const=1 gaddr=4295168000 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=63 is_in=0 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=32 gdma_id=19 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=32 gdma_id=20 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=32 gdma_id=21 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=32 gdma_id=22 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=32 gdma_id=23 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=33 gdma_id=22 bd_func=3
[bmprofile] bd cmd_id bd_id=34 gdma_id=22 bd_func=3
[bmprofile] bd cmd_id bd_id=35 gdma_id=22 bd_func=3
[bmprofile] bd cmd_id bd_id=36 gdma_id=22 bd_func=3
[bmprofile] bd cmd_id bd_id=37 gdma_id=22 bd_func=3
[bmprofile] bd cmd_id bd_id=38 gdma_id=22 bd_func=5
[bmprofile] bd cmd_id bd_id=39 gdma_id=22 bd_func=2
[bmprofile] bd cmd_id bd_id=40 gdma_id=22 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=40 gdma_id=24 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=40 gdma_id=25 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=41 gdma_id=23 bd_func=3
[bmprofile] bd cmd_id bd_id=42 gdma_id=23 bd_func=3
[bmprofile] bd cmd_id bd_id=43 gdma_id=23 bd_func=3
[bmprofile] bd cmd_id bd_id=44 gdma_id=23 bd_func=3
[bmprofile] bd cmd_id bd_id=45 gdma_id=23 bd_func=3
[bmprofile] bd cmd_id bd_id=46 gdma_id=23 bd_func=2
[bmprofile] bd cmd_id bd_id=47 gdma_id=23 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=47 gdma_id=26 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=47 gdma_id=27 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=48 gdma_id=25 bd_func=3
[bmprofile] bd cmd_id bd_id=49 gdma_id=25 bd_func=3
[bmprofile] bd cmd_id bd_id=50 gdma_id=25 bd_func=3
[bmprofile] bd cmd_id bd_id=51 gdma_id=25 bd_func=3
[bmprofile] bd cmd_id bd_id=52 gdma_id=25 bd_func=3
[bmprofile] bd cmd_id bd_id=53 gdma_id=25 bd_func=2
[bmprofile] bd cmd_id bd_id=54 gdma_id=25 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=54 gdma_id=28 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=54 gdma_id=29 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=55 gdma_id=27 bd_func=3
[bmprofile] bd cmd_id bd_id=56 gdma_id=27 bd_func=3
[bmprofile] bd cmd_id bd_id=57 gdma_id=27 bd_func=3
[bmprofile] bd cmd_id bd_id=58 gdma_id=27 bd_func=3
[bmprofile] bd cmd_id bd_id=59 gdma_id=27 bd_func=3
[bmprofile] bd cmd_id bd_id=60 gdma_id=27 bd_func=2
[bmprofile] bd cmd_id bd_id=61 gdma_id=27 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=61 gdma_id=30 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=61 gdma_id=31 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=62 gdma_id=29 bd_func=3
[bmprofile] bd cmd_id bd_id=63 gdma_id=29 bd_func=3
[bmprofile] bd cmd_id bd_id=64 gdma_id=29 bd_func=3
[bmprofile] bd cmd_id bd_id=65 gdma_id=29 bd_func=3
[bmprofile] bd cmd_id bd_id=66 gdma_id=29 bd_func=3
[bmprofile] bd cmd_id bd_id=67 gdma_id=29 bd_func=2
[bmprofile] bd cmd_id bd_id=68 gdma_id=29 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=68 gdma_id=32 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=69 gdma_id=31 bd_func=3
[bmprofile] bd cmd_id bd_id=70 gdma_id=31 bd_func=3
[bmprofile] bd cmd_id bd_id=71 gdma_id=31 bd_func=3
[bmprofile] bd cmd_id bd_id=72 gdma_id=31 bd_func=3
[bmprofile] bd cmd_id bd_id=73 gdma_id=31 bd_func=3
[bmprofile] bd cmd_id bd_id=74 gdma_id=31 bd_func=2
[bmprofile] bd cmd_id bd_id=75 gdma_id=31 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=75 gdma_id=33 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=64 layer_type=Reshape layer_name=
[bmprofile] tensor_id=63 is_in=1 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=64 is_in=0 shape=[1x6144x1] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=65 layer_type=ConcatSlice layer_name=
[bmprofile] tensor_id=-2 is_in=1 shape=[1x6144x4] dtype=8 is_const=0 gaddr=4325941248 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=64 is_in=1 shape=[1x6144x1] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=65 is_in=0 shape=[1x6144x4] dtype=8 is_const=0 gaddr=4325941248 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=75 gdma_id=34 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=75 gdma_id=35 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=76 gdma_id=35 bd_func=3
[bmprofile] bd cmd_id bd_id=77 gdma_id=35 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=77 gdma_id=36 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=66 layer_type=Mul layer_name=
[bmprofile] tensor_id=65 is_in=1 shape=[1x6144x4] dtype=8 is_const=0 gaddr=4325941248 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=18 is_in=1 shape=[1x6144x4] dtype=8 is_const=1 gaddr=4294975488 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=66 is_in=0 shape=[1x6144x4] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=77 gdma_id=37 gdma_dir=0 gdma_func=1
[bmprofile] gdma cmd_id bd_id=77 gdma_id=38 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=77 gdma_id=39 gdma_dir=0 gdma_func=1
[bmprofile] gdma cmd_id bd_id=77 gdma_id=40 gdma_dir=0 gdma_func=1
[bmprofile] bd cmd_id bd_id=78 gdma_id=38 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=78 gdma_id=41 gdma_dir=0 gdma_func=1
[bmprofile] bd cmd_id bd_id=79 gdma_id=40 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=79 gdma_id=42 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=67 layer_type=Reduce layer_name=
[bmprofile] tensor_id=66 is_in=1 shape=[1x6144x4] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=67 is_in=0 shape=[1x6144] dtype=8 is_const=0 gaddr=4326563840 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=79 gdma_id=43 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=80 gdma_id=43 bd_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=80 gdma_id=44 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=68 layer_type=Cast layer_name=
[bmprofile] tensor_id=67 is_in=1 shape=[1x6144] dtype=8 is_const=0 gaddr=4326563840 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=68 is_in=0 shape=[1x6144] dtype=0 is_const=0 gaddr=4326539264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=80 gdma_id=45 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=81 gdma_id=45 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=81 gdma_id=46 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=70 layer_type=Load layer_name=
[bmprofile] tensor_id=60 is_in=1 shape=[1x1x16] dtype=0 is_const=0 gaddr=4326584320 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=70 is_in=0 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=81 gdma_id=47 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=72 layer_type=Active layer_name=
[bmprofile] tensor_id=70 is_in=1 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=72 is_in=0 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=82 gdma_id=47 bd_func=5
[bmprofile] bd cmd_id bd_id=83 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=84 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=85 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=86 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=87 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=88 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=89 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=90 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=91 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=92 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=93 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=94 gdma_id=47 bd_func=9
[bmprofile] bd cmd_id bd_id=95 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=96 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=97 gdma_id=47 bd_func=3
[bmprofile] local_layer: layer_id=71 layer_type=Load layer_name=
[bmprofile] tensor_id=57 is_in=1 shape=[1x1x16] dtype=0 is_const=0 gaddr=4326588416 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=71 is_in=0 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=81 gdma_id=48 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=73 layer_type=Cast layer_name=
[bmprofile] tensor_id=72 is_in=1 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=73 is_in=0 shape=[1x1x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=128 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=98 gdma_id=48 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=76 layer_type=Active layer_name=
[bmprofile] tensor_id=71 is_in=1 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=76 is_in=0 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=99 gdma_id=48 bd_func=5
[bmprofile] bd cmd_id bd_id=100 gdma_id=48 bd_func=5
[bmprofile] bd cmd_id bd_id=101 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=102 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=103 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=104 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=105 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=106 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=107 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=108 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=109 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=110 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=111 gdma_id=48 bd_func=9
[bmprofile] bd cmd_id bd_id=112 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=113 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=114 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=115 gdma_id=48 bd_func=9
[bmprofile] bd cmd_id bd_id=116 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=117 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=118 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=119 gdma_id=48 bd_func=9
[bmprofile] bd cmd_id bd_id=120 gdma_id=48 bd_func=13
[bmprofile] bd cmd_id bd_id=121 gdma_id=48 bd_func=9
[bmprofile] bd cmd_id bd_id=122 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=123 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=124 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=125 gdma_id=48 bd_func=3
[bmprofile] local_layer: layer_id=75 layer_type=Store layer_name=
[bmprofile] tensor_id=73 is_in=1 shape=[1x1x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=128 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=75 is_in=0 shape=[1x1x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=128 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=98 gdma_id=49 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=74 layer_type=Load layer_name=
[bmprofile] tensor_id=17 is_in=1 shape=[1x1x16] dtype=8 is_const=1 gaddr=4294971392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=74 is_in=0 shape=[1x1x16] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=64 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=98 gdma_id=50 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=77 layer_type=Cast layer_name=
[bmprofile] tensor_id=76 is_in=1 shape=[1x1x16] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=77 is_in=0 shape=[1x1x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=126 gdma_id=50 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=78 layer_type=Mul layer_name=
[bmprofile] tensor_id=77 is_in=1 shape=[1x1x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=74 is_in=1 shape=[1x1x16] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=64 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=78 is_in=0 shape=[1x1x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] bd cmd_id bd_id=127 gdma_id=50 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=79 layer_type=Store layer_name=
[bmprofile] tensor_id=78 is_in=1 shape=[1x1x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] tensor_id=79 is_in=0 shape=[1x1x16] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=16 l2addr=0
[bmprofile] gdma cmd_id bd_id=127 gdma_id=51 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=82 layer_type=Active layer_name=
[bmprofile] tensor_id=68 is_in=1 shape=[1x6144] dtype=0 is_const=0 gaddr=4326539264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=82 is_in=0 shape=[1x6144] dtype=0 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=127 gdma_id=52 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=128 gdma_id=52 bd_func=5
[bmprofile] bd cmd_id bd_id=129 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=130 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=131 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=132 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=133 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=134 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=135 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=136 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=137 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=138 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=139 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=140 gdma_id=52 bd_func=9
[bmprofile] bd cmd_id bd_id=141 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=142 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=143 gdma_id=52 bd_func=3
[bmprofile] bd cmd_id bd_id=144 gdma_id=52 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=144 gdma_id=53 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=83 layer_type=Cast layer_name=
[bmprofile] tensor_id=82 is_in=1 shape=[1x6144] dtype=0 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=83 is_in=0 shape=[1x6144] dtype=8 is_const=0 gaddr=4326539264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=144 gdma_id=54 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=145 gdma_id=54 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=145 gdma_id=55 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=84 layer_type=Slice layer_name=
[bmprofile] tensor_id=83 is_in=1 shape=[1x6144] dtype=8 is_const=0 gaddr=4326539264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=84 is_in=0 shape=[1x2048] dtype=8 is_const=0 gaddr=4326539264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=85 layer_type=Slice layer_name=
[bmprofile] tensor_id=83 is_in=1 shape=[1x6144] dtype=8 is_const=0 gaddr=4326539264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=85 is_in=0 shape=[1x2048] dtype=8 is_const=0 gaddr=4326543360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=86 layer_type=Slice layer_name=
[bmprofile] tensor_id=83 is_in=1 shape=[1x6144] dtype=8 is_const=0 gaddr=4326539264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=86 is_in=0 shape=[1x2048] dtype=8 is_const=0 gaddr=4326547456 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=87 layer_type=RecurrentGatedDeltaRule layer_name=
[bmprofile] tensor_id=84 is_in=1 shape=[1x2048] dtype=8 is_const=0 gaddr=4326539264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=85 is_in=1 shape=[1x2048] dtype=8 is_const=0 gaddr=4326543360 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=86 is_in=1 shape=[1x2048] dtype=8 is_const=0 gaddr=4326547456 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=79 is_in=1 shape=[1x1x16] dtype=8 is_const=0 gaddr=4326563840 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=75 is_in=1 shape=[1x1x16] dtype=8 is_const=0 gaddr=4326567936 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=-3 is_in=1 shape=[1x16x128x128] dtype=8 is_const=0 gaddr=4325990400 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=87 is_in=0 shape=[1x1x16x128] dtype=8 is_const=0 gaddr=4326531072 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] bd cmd_id bd_id=146 gdma_id=55 bd_func=5
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=146 gdma_id=56 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=146 gdma_id=57 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=146 gdma_id=58 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=146 gdma_id=59 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=146 gdma_id=60 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=146 gdma_id=61 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=147 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=148 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=149 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=150 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=151 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=152 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=153 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=154 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=155 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=156 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=157 gdma_id=61 bd_func=9
[bmprofile] bd cmd_id bd_id=158 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=159 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=160 gdma_id=61 bd_func=1
[bmprofile] bd cmd_id bd_id=161 gdma_id=61 bd_func=1
[bmprofile] bd cmd_id bd_id=162 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=163 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=164 gdma_id=61 bd_func=9
[bmprofile] bd cmd_id bd_id=165 gdma_id=61 bd_func=13
[bmprofile] bd cmd_id bd_id=166 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=167 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=168 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=169 gdma_id=61 bd_func=1
[bmprofile] bd cmd_id bd_id=170 gdma_id=61 bd_func=1
[bmprofile] bd cmd_id bd_id=171 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=172 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=173 gdma_id=61 bd_func=9
[bmprofile] bd cmd_id bd_id=174 gdma_id=61 bd_func=13
[bmprofile] bd cmd_id bd_id=175 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=176 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=177 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=178 gdma_id=61 bd_func=5
[bmprofile] bd cmd_id bd_id=179 gdma_id=61 bd_func=5
[bmprofile] bd cmd_id bd_id=180 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=181 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=182 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=183 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=184 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=185 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=186 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=187 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=188 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=189 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=190 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=191 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=192 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=193 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=194 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=195 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=196 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=197 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=198 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=199 gdma_id=61 bd_func=3
[bmprofile] bd cmd_id bd_id=200 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=201 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=202 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=203 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=204 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=205 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=206 gdma_id=61 bd_func=2
[bmprofile] bd cmd_id bd_id=207 gdma_id=61 bd_func=2
[bmprofile] gdma cmd_id bd_id=146 gdma_id=62 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=146 gdma_id=63 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=146 gdma_id=64 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=146 gdma_id=65 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=146 gdma_id=66 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=146 gdma_id=67 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=207 gdma_id=68 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=207 gdma_id=69 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=208 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=209 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=210 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=211 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=212 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=213 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=214 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=215 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=216 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=217 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=218 gdma_id=67 bd_func=9
[bmprofile] bd cmd_id bd_id=219 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=220 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=221 gdma_id=67 bd_func=1
[bmprofile] bd cmd_id bd_id=222 gdma_id=67 bd_func=1
[bmprofile] bd cmd_id bd_id=223 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=224 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=225 gdma_id=67 bd_func=9
[bmprofile] bd cmd_id bd_id=226 gdma_id=67 bd_func=13
[bmprofile] bd cmd_id bd_id=227 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=228 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=229 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=230 gdma_id=67 bd_func=1
[bmprofile] bd cmd_id bd_id=231 gdma_id=67 bd_func=1
[bmprofile] bd cmd_id bd_id=232 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=233 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=234 gdma_id=67 bd_func=9
[bmprofile] bd cmd_id bd_id=235 gdma_id=67 bd_func=13
[bmprofile] bd cmd_id bd_id=236 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=237 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=238 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=239 gdma_id=67 bd_func=5
[bmprofile] bd cmd_id bd_id=240 gdma_id=67 bd_func=5
[bmprofile] bd cmd_id bd_id=241 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=242 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=243 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=244 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=245 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=246 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=247 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=248 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=249 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=250 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=251 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=252 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=253 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=254 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=255 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=256 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=257 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=258 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=259 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=260 gdma_id=67 bd_func=3
[bmprofile] bd cmd_id bd_id=261 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=262 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=263 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=264 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=265 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=266 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=267 gdma_id=67 bd_func=2
[bmprofile] bd cmd_id bd_id=268 gdma_id=67 bd_func=2
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=268 gdma_id=70 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=268 gdma_id=71 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=88 layer_type=RMSNorm layer_name=
[bmprofile] tensor_id=87 is_in=1 shape=[1x1x16x128] dtype=8 is_const=0 gaddr=4326531072 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=29 is_in=1 shape=[1x1x1x128] dtype=8 is_const=1 gaddr=4303949824 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=88 is_in=0 shape=[1x1x16x128] dtype=8 is_const=0 gaddr=4326526976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=268 gdma_id=72 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=268 gdma_id=73 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=269 gdma_id=73 bd_func=3
[bmprofile] bd cmd_id bd_id=270 gdma_id=73 bd_func=3
[bmprofile] bd cmd_id bd_id=271 gdma_id=73 bd_func=1
[bmprofile] bd cmd_id bd_id=272 gdma_id=73 bd_func=3
[bmprofile] bd cmd_id bd_id=273 gdma_id=73 bd_func=9
[bmprofile] bd cmd_id bd_id=274 gdma_id=73 bd_func=13
[bmprofile] bd cmd_id bd_id=275 gdma_id=73 bd_func=3
[bmprofile] bd cmd_id bd_id=276 gdma_id=73 bd_func=3
[bmprofile] bd cmd_id bd_id=277 gdma_id=73 bd_func=5
[bmprofile] bd cmd_id bd_id=278 gdma_id=73 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=278 gdma_id=74 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=89 layer_type=Reshape layer_name=
[bmprofile] tensor_id=88 is_in=1 shape=[1x1x16x128] dtype=8 is_const=0 gaddr=4326526976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=89 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326526976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=90 layer_type=Active layer_name=
[bmprofile] tensor_id=49 is_in=1 shape=[1x1x2048] dtype=0 is_const=0 gaddr=4326576128 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=90 is_in=0 shape=[1x1x2048] dtype=0 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=278 gdma_id=75 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=279 gdma_id=75 bd_func=5
[bmprofile] bd cmd_id bd_id=280 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=281 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=282 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=283 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=284 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=285 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=286 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=287 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=288 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=289 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=290 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=291 gdma_id=75 bd_func=9
[bmprofile] bd cmd_id bd_id=292 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=293 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=294 gdma_id=75 bd_func=3
[bmprofile] bd cmd_id bd_id=295 gdma_id=75 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=295 gdma_id=76 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=92 layer_type=Load layer_name=
[bmprofile] tensor_id=90 is_in=1 shape=[1x1x2048] dtype=0 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=92 is_in=0 shape=[1x1x2048] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=295 gdma_id=77 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=94 layer_type=Cast layer_name=
[bmprofile] tensor_id=92 is_in=1 shape=[1x1x2048] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=94 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=2048 l2addr=0
[bmprofile] bd cmd_id bd_id=296 gdma_id=77 bd_func=3
[bmprofile] local_layer: layer_id=93 layer_type=Load layer_name=
[bmprofile] tensor_id=89 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326526976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=93 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=295 gdma_id=78 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=95 layer_type=Mul layer_name=
[bmprofile] tensor_id=93 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=94 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=95 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2048 l2addr=0
[bmprofile] bd cmd_id bd_id=297 gdma_id=78 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=96 layer_type=Store layer_name=
[bmprofile] tensor_id=95 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=96 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=297 gdma_id=79 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=99 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=96 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326522880 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=32 is_in=1 shape=[2048x1024] dtype=3 is_const=1 gaddr=4304052224 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=31 is_in=1 shape=[64x32x16] dtype=8 is_const=1 gaddr=4303986688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=30 is_in=1 shape=[64x32x16] dtype=3 is_const=1 gaddr=4303953920 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=99 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=297 gdma_id=80 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=297 gdma_id=81 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=297 gdma_id=82 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=297 gdma_id=83 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=297 gdma_id=84 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=298 gdma_id=83 bd_func=3
[bmprofile] bd cmd_id bd_id=299 gdma_id=83 bd_func=3
[bmprofile] bd cmd_id bd_id=300 gdma_id=83 bd_func=3
[bmprofile] bd cmd_id bd_id=301 gdma_id=83 bd_func=3
[bmprofile] bd cmd_id bd_id=302 gdma_id=83 bd_func=3
[bmprofile] bd cmd_id bd_id=303 gdma_id=83 bd_func=5
[bmprofile] bd cmd_id bd_id=304 gdma_id=83 bd_func=2
[bmprofile] bd cmd_id bd_id=305 gdma_id=83 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=305 gdma_id=85 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=306 gdma_id=84 bd_func=3
[bmprofile] bd cmd_id bd_id=307 gdma_id=84 bd_func=3
[bmprofile] bd cmd_id bd_id=308 gdma_id=84 bd_func=3
[bmprofile] bd cmd_id bd_id=309 gdma_id=84 bd_func=3
[bmprofile] bd cmd_id bd_id=310 gdma_id=84 bd_func=3
[bmprofile] bd cmd_id bd_id=311 gdma_id=84 bd_func=2
[bmprofile] bd cmd_id bd_id=312 gdma_id=84 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=312 gdma_id=86 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=101 layer_type=Load layer_name=
[bmprofile] tensor_id=-4 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4325937152 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=101 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=312 gdma_id=87 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=102 layer_type=Load layer_name=
[bmprofile] tensor_id=99 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=102 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=312 gdma_id=88 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=104 layer_type=Add layer_name=
[bmprofile] tensor_id=101 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=102 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=104 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] bd cmd_id bd_id=313 gdma_id=88 bd_func=3
[bmprofile] local_layer: layer_id=103 layer_type=Load layer_name=
[bmprofile] tensor_id=45 is_in=1 shape=[1x1x2048] dtype=8 is_const=1 gaddr=4325933056 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=103 is_in=0 shape=[1x1x2048] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=312 gdma_id=89 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=106 layer_type=RMSNorm layer_name=
[bmprofile] tensor_id=104 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=103 is_in=1 shape=[1x1x2048] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=106 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=2048 l2addr=0
[bmprofile] bd cmd_id bd_id=314 gdma_id=89 bd_func=3
[bmprofile] bd cmd_id bd_id=315 gdma_id=89 bd_func=3
[bmprofile] bd cmd_id bd_id=316 gdma_id=89 bd_func=1
[bmprofile] bd cmd_id bd_id=317 gdma_id=89 bd_func=1
[bmprofile] bd cmd_id bd_id=318 gdma_id=89 bd_func=3
[bmprofile] bd cmd_id bd_id=319 gdma_id=89 bd_func=9
[bmprofile] bd cmd_id bd_id=320 gdma_id=89 bd_func=13
[bmprofile] bd cmd_id bd_id=321 gdma_id=89 bd_func=3
[bmprofile] bd cmd_id bd_id=322 gdma_id=89 bd_func=3
[bmprofile] bd cmd_id bd_id=323 gdma_id=89 bd_func=3
[bmprofile] bd cmd_id bd_id=324 gdma_id=89 bd_func=5
[bmprofile] bd cmd_id bd_id=325 gdma_id=89 bd_func=3
[bmprofile] local_layer: layer_id=105 layer_type=Store layer_name=
[bmprofile] tensor_id=104 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=105 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=313 gdma_id=90 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=107 layer_type=Store layer_name=
[bmprofile] tensor_id=106 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=107 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=325 gdma_id=91 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=110 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=107 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326567936 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=41 is_in=1 shape=[6144x1024] dtype=3 is_const=1 gaddr=4313055232 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=40 is_in=1 shape=[64x96x16] dtype=8 is_const=1 gaddr=4312858624 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=39 is_in=1 shape=[64x96x16] dtype=3 is_const=1 gaddr=4312760320 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=110 is_in=0 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=325 gdma_id=92 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=325 gdma_id=93 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=325 gdma_id=94 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=325 gdma_id=95 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=325 gdma_id=96 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=326 gdma_id=95 bd_func=3
[bmprofile] bd cmd_id bd_id=327 gdma_id=95 bd_func=3
[bmprofile] bd cmd_id bd_id=328 gdma_id=95 bd_func=3
[bmprofile] bd cmd_id bd_id=329 gdma_id=95 bd_func=3
[bmprofile] bd cmd_id bd_id=330 gdma_id=95 bd_func=3
[bmprofile] bd cmd_id bd_id=331 gdma_id=95 bd_func=5
[bmprofile] bd cmd_id bd_id=332 gdma_id=95 bd_func=2
[bmprofile] bd cmd_id bd_id=333 gdma_id=95 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=333 gdma_id=97 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=333 gdma_id=98 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=334 gdma_id=96 bd_func=3
[bmprofile] bd cmd_id bd_id=335 gdma_id=96 bd_func=3
[bmprofile] bd cmd_id bd_id=336 gdma_id=96 bd_func=3
[bmprofile] bd cmd_id bd_id=337 gdma_id=96 bd_func=3
[bmprofile] bd cmd_id bd_id=338 gdma_id=96 bd_func=3
[bmprofile] bd cmd_id bd_id=339 gdma_id=96 bd_func=2
[bmprofile] bd cmd_id bd_id=340 gdma_id=96 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=340 gdma_id=99 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=340 gdma_id=100 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=341 gdma_id=98 bd_func=3
[bmprofile] bd cmd_id bd_id=342 gdma_id=98 bd_func=3
[bmprofile] bd cmd_id bd_id=343 gdma_id=98 bd_func=3
[bmprofile] bd cmd_id bd_id=344 gdma_id=98 bd_func=3
[bmprofile] bd cmd_id bd_id=345 gdma_id=98 bd_func=3
[bmprofile] bd cmd_id bd_id=346 gdma_id=98 bd_func=2
[bmprofile] bd cmd_id bd_id=347 gdma_id=98 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=347 gdma_id=101 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=347 gdma_id=102 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=348 gdma_id=100 bd_func=3
[bmprofile] bd cmd_id bd_id=349 gdma_id=100 bd_func=3
[bmprofile] bd cmd_id bd_id=350 gdma_id=100 bd_func=3
[bmprofile] bd cmd_id bd_id=351 gdma_id=100 bd_func=3
[bmprofile] bd cmd_id bd_id=352 gdma_id=100 bd_func=3
[bmprofile] bd cmd_id bd_id=353 gdma_id=100 bd_func=2
[bmprofile] bd cmd_id bd_id=354 gdma_id=100 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=354 gdma_id=103 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=354 gdma_id=104 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=355 gdma_id=102 bd_func=3
[bmprofile] bd cmd_id bd_id=356 gdma_id=102 bd_func=3
[bmprofile] bd cmd_id bd_id=357 gdma_id=102 bd_func=3
[bmprofile] bd cmd_id bd_id=358 gdma_id=102 bd_func=3
[bmprofile] bd cmd_id bd_id=359 gdma_id=102 bd_func=3
[bmprofile] bd cmd_id bd_id=360 gdma_id=102 bd_func=2
[bmprofile] bd cmd_id bd_id=361 gdma_id=102 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=361 gdma_id=105 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=362 gdma_id=104 bd_func=3
[bmprofile] bd cmd_id bd_id=363 gdma_id=104 bd_func=3
[bmprofile] bd cmd_id bd_id=364 gdma_id=104 bd_func=3
[bmprofile] bd cmd_id bd_id=365 gdma_id=104 bd_func=3
[bmprofile] bd cmd_id bd_id=366 gdma_id=104 bd_func=3
[bmprofile] bd cmd_id bd_id=367 gdma_id=104 bd_func=2
[bmprofile] bd cmd_id bd_id=368 gdma_id=104 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=368 gdma_id=106 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=111 layer_type=Cast layer_name=
[bmprofile] tensor_id=110 is_in=1 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=111 is_in=0 shape=[1x1x6144] dtype=0 is_const=0 gaddr=4326539264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=368 gdma_id=107 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=369 gdma_id=107 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=369 gdma_id=108 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=112 layer_type=Active layer_name=
[bmprofile] tensor_id=111 is_in=1 shape=[1x1x6144] dtype=0 is_const=0 gaddr=4326539264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=112 is_in=0 shape=[1x1x6144] dtype=0 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=369 gdma_id=109 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=370 gdma_id=109 bd_func=5
[bmprofile] bd cmd_id bd_id=371 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=372 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=373 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=374 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=375 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=376 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=377 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=378 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=379 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=380 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=381 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=382 gdma_id=109 bd_func=9
[bmprofile] bd cmd_id bd_id=383 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=384 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=385 gdma_id=109 bd_func=3
[bmprofile] bd cmd_id bd_id=386 gdma_id=109 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=386 gdma_id=110 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=113 layer_type=Cast layer_name=
[bmprofile] tensor_id=112 is_in=1 shape=[1x1x6144] dtype=0 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=113 is_in=0 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326539264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=386 gdma_id=111 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=387 gdma_id=111 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=387 gdma_id=112 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=114 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=107 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326567936 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=44 is_in=1 shape=[6144x1024] dtype=3 is_const=1 gaddr=4319641600 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=43 is_in=1 shape=[64x96x16] dtype=8 is_const=1 gaddr=4319444992 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=42 is_in=1 shape=[64x96x16] dtype=3 is_const=1 gaddr=4319346688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=114 is_in=0 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326526976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=387 gdma_id=113 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=387 gdma_id=114 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=387 gdma_id=115 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=387 gdma_id=116 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=387 gdma_id=117 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=388 gdma_id=116 bd_func=3
[bmprofile] bd cmd_id bd_id=389 gdma_id=116 bd_func=3
[bmprofile] bd cmd_id bd_id=390 gdma_id=116 bd_func=3
[bmprofile] bd cmd_id bd_id=391 gdma_id=116 bd_func=3
[bmprofile] bd cmd_id bd_id=392 gdma_id=116 bd_func=3
[bmprofile] bd cmd_id bd_id=393 gdma_id=116 bd_func=5
[bmprofile] bd cmd_id bd_id=394 gdma_id=116 bd_func=2
[bmprofile] bd cmd_id bd_id=395 gdma_id=116 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=395 gdma_id=118 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=395 gdma_id=119 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=396 gdma_id=117 bd_func=3
[bmprofile] bd cmd_id bd_id=397 gdma_id=117 bd_func=3
[bmprofile] bd cmd_id bd_id=398 gdma_id=117 bd_func=3
[bmprofile] bd cmd_id bd_id=399 gdma_id=117 bd_func=3
[bmprofile] bd cmd_id bd_id=400 gdma_id=117 bd_func=3
[bmprofile] bd cmd_id bd_id=401 gdma_id=117 bd_func=2
[bmprofile] bd cmd_id bd_id=402 gdma_id=117 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=402 gdma_id=120 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=402 gdma_id=121 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=403 gdma_id=119 bd_func=3
[bmprofile] bd cmd_id bd_id=404 gdma_id=119 bd_func=3
[bmprofile] bd cmd_id bd_id=405 gdma_id=119 bd_func=3
[bmprofile] bd cmd_id bd_id=406 gdma_id=119 bd_func=3
[bmprofile] bd cmd_id bd_id=407 gdma_id=119 bd_func=3
[bmprofile] bd cmd_id bd_id=408 gdma_id=119 bd_func=2
[bmprofile] bd cmd_id bd_id=409 gdma_id=119 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=409 gdma_id=122 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=409 gdma_id=123 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=410 gdma_id=121 bd_func=3
[bmprofile] bd cmd_id bd_id=411 gdma_id=121 bd_func=3
[bmprofile] bd cmd_id bd_id=412 gdma_id=121 bd_func=3
[bmprofile] bd cmd_id bd_id=413 gdma_id=121 bd_func=3
[bmprofile] bd cmd_id bd_id=414 gdma_id=121 bd_func=3
[bmprofile] bd cmd_id bd_id=415 gdma_id=121 bd_func=2
[bmprofile] bd cmd_id bd_id=416 gdma_id=121 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=416 gdma_id=124 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=416 gdma_id=125 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=417 gdma_id=123 bd_func=3
[bmprofile] bd cmd_id bd_id=418 gdma_id=123 bd_func=3
[bmprofile] bd cmd_id bd_id=419 gdma_id=123 bd_func=3
[bmprofile] bd cmd_id bd_id=420 gdma_id=123 bd_func=3
[bmprofile] bd cmd_id bd_id=421 gdma_id=123 bd_func=3
[bmprofile] bd cmd_id bd_id=422 gdma_id=123 bd_func=2
[bmprofile] bd cmd_id bd_id=423 gdma_id=123 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=423 gdma_id=126 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=424 gdma_id=125 bd_func=3
[bmprofile] bd cmd_id bd_id=425 gdma_id=125 bd_func=3
[bmprofile] bd cmd_id bd_id=426 gdma_id=125 bd_func=3
[bmprofile] bd cmd_id bd_id=427 gdma_id=125 bd_func=3
[bmprofile] bd cmd_id bd_id=428 gdma_id=125 bd_func=3
[bmprofile] bd cmd_id bd_id=429 gdma_id=125 bd_func=2
[bmprofile] bd cmd_id bd_id=430 gdma_id=125 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=430 gdma_id=127 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=115 layer_type=Mul layer_name=
[bmprofile] tensor_id=113 is_in=1 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326539264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=114 is_in=1 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326526976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=115 is_in=0 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=430 gdma_id=128 gdma_dir=0 gdma_func=1
[bmprofile] gdma cmd_id bd_id=430 gdma_id=129 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=431 gdma_id=129 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=431 gdma_id=130 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=116 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=115 is_in=1 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326514688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=38 is_in=1 shape=[2048x3072] dtype=3 is_const=1 gaddr=4306468864 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=37 is_in=1 shape=[64x32x48] dtype=8 is_const=1 gaddr=4306272256 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=36 is_in=1 shape=[64x32x48] dtype=3 is_const=1 gaddr=4306173952 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=116 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326526976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=431 gdma_id=131 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=431 gdma_id=132 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=431 gdma_id=133 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=431 gdma_id=134 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=431 gdma_id=135 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=432 gdma_id=134 bd_func=3
[bmprofile] bd cmd_id bd_id=433 gdma_id=134 bd_func=3
[bmprofile] bd cmd_id bd_id=434 gdma_id=134 bd_func=3
[bmprofile] bd cmd_id bd_id=435 gdma_id=134 bd_func=3
[bmprofile] bd cmd_id bd_id=436 gdma_id=134 bd_func=3
[bmprofile] bd cmd_id bd_id=437 gdma_id=134 bd_func=5
[bmprofile] bd cmd_id bd_id=438 gdma_id=134 bd_func=2
[bmprofile] bd cmd_id bd_id=439 gdma_id=134 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=439 gdma_id=136 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=439 gdma_id=137 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=440 gdma_id=135 bd_func=3
[bmprofile] bd cmd_id bd_id=441 gdma_id=135 bd_func=3
[bmprofile] bd cmd_id bd_id=442 gdma_id=135 bd_func=3
[bmprofile] bd cmd_id bd_id=443 gdma_id=135 bd_func=3
[bmprofile] bd cmd_id bd_id=444 gdma_id=135 bd_func=3
[bmprofile] bd cmd_id bd_id=445 gdma_id=135 bd_func=2
[bmprofile] bd cmd_id bd_id=446 gdma_id=135 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=446 gdma_id=138 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=446 gdma_id=139 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=447 gdma_id=137 bd_func=3
[bmprofile] bd cmd_id bd_id=448 gdma_id=137 bd_func=3
[bmprofile] bd cmd_id bd_id=449 gdma_id=137 bd_func=3
[bmprofile] bd cmd_id bd_id=450 gdma_id=137 bd_func=3
[bmprofile] bd cmd_id bd_id=451 gdma_id=137 bd_func=3
[bmprofile] bd cmd_id bd_id=452 gdma_id=137 bd_func=2
[bmprofile] bd cmd_id bd_id=453 gdma_id=137 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=453 gdma_id=140 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=453 gdma_id=141 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=454 gdma_id=139 bd_func=3
[bmprofile] bd cmd_id bd_id=455 gdma_id=139 bd_func=3
[bmprofile] bd cmd_id bd_id=456 gdma_id=139 bd_func=3
[bmprofile] bd cmd_id bd_id=457 gdma_id=139 bd_func=3
[bmprofile] bd cmd_id bd_id=458 gdma_id=139 bd_func=3
[bmprofile] bd cmd_id bd_id=459 gdma_id=139 bd_func=2
[bmprofile] bd cmd_id bd_id=460 gdma_id=139 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=460 gdma_id=142 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=460 gdma_id=143 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=461 gdma_id=141 bd_func=3
[bmprofile] bd cmd_id bd_id=462 gdma_id=141 bd_func=3
[bmprofile] bd cmd_id bd_id=463 gdma_id=141 bd_func=3
[bmprofile] bd cmd_id bd_id=464 gdma_id=141 bd_func=3
[bmprofile] bd cmd_id bd_id=465 gdma_id=141 bd_func=3
[bmprofile] bd cmd_id bd_id=466 gdma_id=141 bd_func=2
[bmprofile] bd cmd_id bd_id=467 gdma_id=141 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=467 gdma_id=144 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=467 gdma_id=145 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=468 gdma_id=143 bd_func=3
[bmprofile] bd cmd_id bd_id=469 gdma_id=143 bd_func=3
[bmprofile] bd cmd_id bd_id=470 gdma_id=143 bd_func=3
[bmprofile] bd cmd_id bd_id=471 gdma_id=143 bd_func=3
[bmprofile] bd cmd_id bd_id=472 gdma_id=143 bd_func=3
[bmprofile] bd cmd_id bd_id=473 gdma_id=143 bd_func=2
[bmprofile] bd cmd_id bd_id=474 gdma_id=143 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=474 gdma_id=146 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=475 gdma_id=145 bd_func=3
[bmprofile] bd cmd_id bd_id=476 gdma_id=145 bd_func=3
[bmprofile] bd cmd_id bd_id=477 gdma_id=145 bd_func=3
[bmprofile] bd cmd_id bd_id=478 gdma_id=145 bd_func=3
[bmprofile] bd cmd_id bd_id=479 gdma_id=145 bd_func=3
[bmprofile] bd cmd_id bd_id=480 gdma_id=145 bd_func=2
[bmprofile] bd cmd_id bd_id=481 gdma_id=145 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=481 gdma_id=147 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=117 layer_type=Add layer_name=
[bmprofile] tensor_id=105 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326563840 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=116 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326526976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=117 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4325937152 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=481 gdma_id=148 gdma_dir=0 gdma_func=1
[bmprofile] gdma cmd_id bd_id=481 gdma_id=149 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=482 gdma_id=149 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=482 gdma_id=150 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] insert bd end (cmd_id bd_id=483)
[bmprofile] bd cmd_id bd_id=483 gdma_id=0 bd_func=15
[bmprofile] insert gdma end (cmd_id gdma_id=151)
[bmprofile] gdma cmd_id bd_id=0 gdma_id=151 gdma_dir=0 gdma_func=6
[bmprofile] end to run subnet_id=0
