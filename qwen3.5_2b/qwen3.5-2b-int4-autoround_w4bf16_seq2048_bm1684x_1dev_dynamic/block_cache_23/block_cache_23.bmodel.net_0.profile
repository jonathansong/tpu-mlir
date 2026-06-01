[bmprofile] is_mlir=1
...Start Profile Log...
[bmprofile] start to run subnet_id=0

[bmprofile] global_layer: layer_id=50 layer_type=RMSNorm layer_name=
[bmprofile] tensor_id=-1 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4322701312 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=20 is_in=1 shape=[1x1x2048] dtype=8 is_const=1 gaddr=4294967296 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=50 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
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

[bmprofile] global_layer: layer_id=51 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=50 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=41 is_in=1 shape=[4096x1024] dtype=3 is_const=1 gaddr=4317683712 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=40 is_in=1 shape=[64x64x16] dtype=8 is_const=1 gaddr=4317552640 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=39 is_in=1 shape=[64x64x16] dtype=3 is_const=1 gaddr=4317487104 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=51 is_in=0 shape=[1x1x4096] dtype=8 is_const=0 gaddr=4326928384 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
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
[bmprofile] gdma cmd_id bd_id=18 gdma_id=10 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=19 gdma_id=8 bd_func=3
[bmprofile] bd cmd_id bd_id=20 gdma_id=8 bd_func=3
[bmprofile] bd cmd_id bd_id=21 gdma_id=8 bd_func=3
[bmprofile] bd cmd_id bd_id=22 gdma_id=8 bd_func=3
[bmprofile] bd cmd_id bd_id=23 gdma_id=8 bd_func=3
[bmprofile] bd cmd_id bd_id=24 gdma_id=8 bd_func=2
[bmprofile] bd cmd_id bd_id=25 gdma_id=8 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=25 gdma_id=11 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=25 gdma_id=12 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=26 gdma_id=10 bd_func=3
[bmprofile] bd cmd_id bd_id=27 gdma_id=10 bd_func=3
[bmprofile] bd cmd_id bd_id=28 gdma_id=10 bd_func=3
[bmprofile] bd cmd_id bd_id=29 gdma_id=10 bd_func=3
[bmprofile] bd cmd_id bd_id=30 gdma_id=10 bd_func=3
[bmprofile] bd cmd_id bd_id=31 gdma_id=10 bd_func=2
[bmprofile] bd cmd_id bd_id=32 gdma_id=10 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=32 gdma_id=13 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=33 gdma_id=12 bd_func=3
[bmprofile] bd cmd_id bd_id=34 gdma_id=12 bd_func=3
[bmprofile] bd cmd_id bd_id=35 gdma_id=12 bd_func=3
[bmprofile] bd cmd_id bd_id=36 gdma_id=12 bd_func=3
[bmprofile] bd cmd_id bd_id=37 gdma_id=12 bd_func=3
[bmprofile] bd cmd_id bd_id=38 gdma_id=12 bd_func=2
[bmprofile] bd cmd_id bd_id=39 gdma_id=12 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=39 gdma_id=14 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=52 layer_type=Reshape layer_name=
[bmprofile] tensor_id=51 is_in=1 shape=[1x1x4096] dtype=8 is_const=0 gaddr=4326928384 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=52 is_in=0 shape=[1x1x8x512] dtype=8 is_const=0 gaddr=4326928384 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=54 layer_type=Load layer_name=
[bmprofile] tensor_id=52 is_in=1 shape=[1x1x8x512] dtype=8 is_const=0 gaddr=4326928384 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=54 is_in=0 shape=[1x1x8x512] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=39 gdma_id=15 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=55 layer_type=Slice layer_name=
[bmprofile] tensor_id=54 is_in=1 shape=[1x1x8x512] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=55 is_in=0 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=40 gdma_id=15 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=57 layer_type=Slice layer_name=
[bmprofile] tensor_id=54 is_in=1 shape=[1x1x8x512] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=57 is_in=0 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=41 gdma_id=15 bd_func=3
[bmprofile] local_layer: layer_id=56 layer_type=Store layer_name=
[bmprofile] tensor_id=55 is_in=1 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=56 is_in=0 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=40 gdma_id=16 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=58 layer_type=Cast layer_name=
[bmprofile] tensor_id=57 is_in=1 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=58 is_in=0 shape=[1x1x8x256] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=42 gdma_id=16 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=59 layer_type=Store layer_name=
[bmprofile] tensor_id=58 is_in=1 shape=[1x1x8x256] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=59 is_in=0 shape=[1x1x8x256] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=42 gdma_id=17 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=62 layer_type=Active layer_name=
[bmprofile] tensor_id=59 is_in=1 shape=[1x1x8x256] dtype=0 is_const=0 gaddr=4326940672 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=62 is_in=0 shape=[1x1x8x256] dtype=0 is_const=0 gaddr=4326928384 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=42 gdma_id=18 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=43 gdma_id=18 bd_func=5
[bmprofile] bd cmd_id bd_id=44 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=45 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=46 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=47 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=48 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=49 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=50 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=51 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=52 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=53 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=54 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=55 gdma_id=18 bd_func=9
[bmprofile] bd cmd_id bd_id=56 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=57 gdma_id=18 bd_func=3
[bmprofile] bd cmd_id bd_id=58 gdma_id=18 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=58 gdma_id=19 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=63 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=50 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=34 is_in=1 shape=[512x1024] dtype=3 is_const=1 gaddr=4314763264 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=33 is_in=1 shape=[64x8x16] dtype=8 is_const=1 gaddr=4314746880 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=32 is_in=1 shape=[64x8x16] dtype=3 is_const=1 gaddr=4314738688 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=63 is_in=0 shape=[1x1x512] dtype=8 is_const=0 gaddr=4326940672 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=58 gdma_id=20 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=58 gdma_id=21 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=58 gdma_id=22 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=58 gdma_id=23 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=59 gdma_id=23 bd_func=3
[bmprofile] bd cmd_id bd_id=60 gdma_id=23 bd_func=3
[bmprofile] bd cmd_id bd_id=61 gdma_id=23 bd_func=3
[bmprofile] bd cmd_id bd_id=62 gdma_id=23 bd_func=3
[bmprofile] bd cmd_id bd_id=63 gdma_id=23 bd_func=3
[bmprofile] bd cmd_id bd_id=64 gdma_id=23 bd_func=5
[bmprofile] bd cmd_id bd_id=65 gdma_id=23 bd_func=2
[bmprofile] bd cmd_id bd_id=66 gdma_id=23 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=66 gdma_id=24 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=64 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=50 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=44 is_in=1 shape=[512x1024] dtype=3 is_const=1 gaddr=4321902592 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=43 is_in=1 shape=[64x8x16] dtype=8 is_const=1 gaddr=4321886208 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=42 is_in=1 shape=[64x8x16] dtype=3 is_const=1 gaddr=4321878016 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=64 is_in=0 shape=[1x1x512] dtype=8 is_const=0 gaddr=4326916096 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=66 gdma_id=25 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=26 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=27 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=66 gdma_id=28 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=67 gdma_id=28 bd_func=3
[bmprofile] bd cmd_id bd_id=68 gdma_id=28 bd_func=3
[bmprofile] bd cmd_id bd_id=69 gdma_id=28 bd_func=3
[bmprofile] bd cmd_id bd_id=70 gdma_id=28 bd_func=3
[bmprofile] bd cmd_id bd_id=71 gdma_id=28 bd_func=3
[bmprofile] bd cmd_id bd_id=72 gdma_id=28 bd_func=5
[bmprofile] bd cmd_id bd_id=73 gdma_id=28 bd_func=2
[bmprofile] bd cmd_id bd_id=74 gdma_id=28 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=74 gdma_id=29 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=65 layer_type=Reshape layer_name=
[bmprofile] tensor_id=63 is_in=1 shape=[1x1x512] dtype=8 is_const=0 gaddr=4326940672 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=65 is_in=0 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=4326940672 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=66 layer_type=Reshape layer_name=
[bmprofile] tensor_id=64 is_in=1 shape=[1x1x512] dtype=8 is_const=0 gaddr=4326916096 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=66 is_in=0 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=4326916096 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=68 layer_type=Load layer_name=
[bmprofile] tensor_id=56 is_in=1 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=68 is_in=0 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=256 l2addr=0
[bmprofile] gdma cmd_id bd_id=74 gdma_id=30 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=69 layer_type=Load layer_name=
[bmprofile] tensor_id=38 is_in=1 shape=[1x1x1x256] dtype=8 is_const=1 gaddr=4317483008 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=69 is_in=0 shape=[1x1x1x256] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=256 l2addr=0
[bmprofile] gdma cmd_id bd_id=74 gdma_id=31 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=72 layer_type=RMSNorm layer_name=
[bmprofile] tensor_id=68 is_in=1 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=256 l2addr=0
[bmprofile] tensor_id=69 is_in=1 shape=[1x1x1x256] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=256 l2addr=0
[bmprofile] tensor_id=72 is_in=0 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=256 l2addr=0
[bmprofile] bd cmd_id bd_id=75 gdma_id=31 bd_func=3
[bmprofile] bd cmd_id bd_id=76 gdma_id=31 bd_func=3
[bmprofile] bd cmd_id bd_id=77 gdma_id=31 bd_func=1
[bmprofile] bd cmd_id bd_id=78 gdma_id=31 bd_func=1
[bmprofile] bd cmd_id bd_id=79 gdma_id=31 bd_func=3
[bmprofile] bd cmd_id bd_id=80 gdma_id=31 bd_func=9
[bmprofile] bd cmd_id bd_id=81 gdma_id=31 bd_func=13
[bmprofile] bd cmd_id bd_id=82 gdma_id=31 bd_func=3
[bmprofile] bd cmd_id bd_id=83 gdma_id=31 bd_func=3
[bmprofile] bd cmd_id bd_id=84 gdma_id=31 bd_func=3
[bmprofile] bd cmd_id bd_id=85 gdma_id=31 bd_func=5
[bmprofile] bd cmd_id bd_id=86 gdma_id=31 bd_func=3
[bmprofile] local_layer: layer_id=70 layer_type=Load layer_name=
[bmprofile] tensor_id=65 is_in=1 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=4326940672 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=70 is_in=0 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=256 l2addr=0
[bmprofile] gdma cmd_id bd_id=74 gdma_id=32 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=71 layer_type=Load layer_name=
[bmprofile] tensor_id=31 is_in=1 shape=[1x1x1x256] dtype=8 is_const=1 gaddr=4314734592 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=71 is_in=0 shape=[1x1x1x256] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=256 l2addr=0
[bmprofile] gdma cmd_id bd_id=74 gdma_id=33 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=74 layer_type=RMSNorm layer_name=
[bmprofile] tensor_id=70 is_in=1 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=256 l2addr=0
[bmprofile] tensor_id=71 is_in=1 shape=[1x1x1x256] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=256 l2addr=0
[bmprofile] tensor_id=74 is_in=0 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=256 l2addr=0
[bmprofile] bd cmd_id bd_id=87 gdma_id=33 bd_func=3
[bmprofile] bd cmd_id bd_id=88 gdma_id=33 bd_func=3
[bmprofile] bd cmd_id bd_id=89 gdma_id=33 bd_func=1
[bmprofile] bd cmd_id bd_id=90 gdma_id=33 bd_func=1
[bmprofile] bd cmd_id bd_id=91 gdma_id=33 bd_func=3
[bmprofile] bd cmd_id bd_id=92 gdma_id=33 bd_func=9
[bmprofile] bd cmd_id bd_id=93 gdma_id=33 bd_func=13
[bmprofile] bd cmd_id bd_id=94 gdma_id=33 bd_func=3
[bmprofile] bd cmd_id bd_id=95 gdma_id=33 bd_func=3
[bmprofile] bd cmd_id bd_id=96 gdma_id=33 bd_func=3
[bmprofile] bd cmd_id bd_id=97 gdma_id=33 bd_func=5
[bmprofile] bd cmd_id bd_id=98 gdma_id=33 bd_func=3
[bmprofile] local_layer: layer_id=73 layer_type=Store layer_name=
[bmprofile] tensor_id=72 is_in=1 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=256 l2addr=0
[bmprofile] tensor_id=73 is_in=0 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=256 l2addr=0
[bmprofile] gdma cmd_id bd_id=86 gdma_id=34 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=75 layer_type=Store layer_name=
[bmprofile] tensor_id=74 is_in=1 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=256 l2addr=0
[bmprofile] tensor_id=75 is_in=0 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=256 l2addr=0
[bmprofile] gdma cmd_id bd_id=98 gdma_id=35 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=78 layer_type=Gather layer_name=
[bmprofile] tensor_id=47 is_in=1 shape=[2048x1x32] dtype=8 is_const=1 gaddr=4322435072 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=-2 is_in=1 shape=[3x1] dtype=6 is_const=0 gaddr=4322705408 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=78 is_in=0 shape=[3x1x1x32] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=98 gdma_id=36 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=99 gdma_id=36 bd_func=3
[bmprofile] bd cmd_id bd_id=100 gdma_id=36 bd_func=13
[bmprofile] gdma cmd_id bd_id=100 gdma_id=37 gdma_dir=0 gdma_func=7

[bmprofile] global_layer: layer_id=79 layer_type=Reshape layer_name=
[bmprofile] tensor_id=78 is_in=1 shape=[3x1x1x32] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=79 is_in=0 shape=[96x1x1] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=80 layer_type=Gather layer_name=
[bmprofile] tensor_id=79 is_in=1 shape=[96x1x1] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=46 is_in=1 shape=[1x64] dtype=6 is_const=1 gaddr=4322430976 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=80 is_in=0 shape=[1x64x1x1] dtype=8 is_const=0 gaddr=4326940672 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=100 gdma_id=38 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=101 gdma_id=38 bd_func=3
[bmprofile] bd cmd_id bd_id=102 gdma_id=38 bd_func=13
[bmprofile] gdma cmd_id bd_id=102 gdma_id=39 gdma_dir=0 gdma_func=7

[bmprofile] global_layer: layer_id=81 layer_type=Reshape layer_name=
[bmprofile] tensor_id=80 is_in=1 shape=[1x64x1x1] dtype=8 is_const=0 gaddr=4326940672 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=81 is_in=0 shape=[1x1x1x64] dtype=8 is_const=0 gaddr=4326940672 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=82 layer_type=Gather layer_name=
[bmprofile] tensor_id=49 is_in=1 shape=[2048x1x32] dtype=8 is_const=1 gaddr=4322570240 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=-3 is_in=1 shape=[3x1] dtype=6 is_const=0 gaddr=4322705408 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=82 is_in=0 shape=[3x1x1x32] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=102 gdma_id=40 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=103 gdma_id=40 bd_func=3
[bmprofile] bd cmd_id bd_id=104 gdma_id=40 bd_func=13
[bmprofile] gdma cmd_id bd_id=104 gdma_id=41 gdma_dir=0 gdma_func=7

[bmprofile] global_layer: layer_id=83 layer_type=Reshape layer_name=
[bmprofile] tensor_id=82 is_in=1 shape=[3x1x1x32] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=83 is_in=0 shape=[96x1x1] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=84 layer_type=Gather layer_name=
[bmprofile] tensor_id=83 is_in=1 shape=[96x1x1] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=48 is_in=1 shape=[1x64] dtype=6 is_const=1 gaddr=4322566144 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=84 is_in=0 shape=[1x64x1x1] dtype=8 is_const=0 gaddr=4326948864 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] gdma cmd_id bd_id=104 gdma_id=42 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=105 gdma_id=42 bd_func=3
[bmprofile] bd cmd_id bd_id=106 gdma_id=42 bd_func=13
[bmprofile] gdma cmd_id bd_id=106 gdma_id=43 gdma_dir=0 gdma_func=7

[bmprofile] global_layer: layer_id=85 layer_type=Reshape layer_name=
[bmprofile] tensor_id=84 is_in=1 shape=[1x64x1x1] dtype=8 is_const=0 gaddr=4326948864 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=85 is_in=0 shape=[1x1x1x64] dtype=8 is_const=0 gaddr=4326948864 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=87 layer_type=Load layer_name=
[bmprofile] tensor_id=73 is_in=1 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=87 is_in=0 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=106 gdma_id=44 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=88 layer_type=Slice layer_name=
[bmprofile] tensor_id=87 is_in=1 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=88 is_in=0 shape=[1x1x8x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=19456 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=107 gdma_id=44 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=91 layer_type=Slice layer_name=
[bmprofile] tensor_id=87 is_in=1 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=91 is_in=0 shape=[1x1x8x192] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=108 gdma_id=44 bd_func=3
[bmprofile] local_layer: layer_id=89 layer_type=Load layer_name=
[bmprofile] tensor_id=85 is_in=1 shape=[1x1x1x64] dtype=8 is_const=0 gaddr=4326948864 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=89 is_in=0 shape=[1x1x1x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=107 gdma_id=45 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=90 layer_type=Load layer_name=
[bmprofile] tensor_id=81 is_in=1 shape=[1x1x1x64] dtype=8 is_const=0 gaddr=4326940672 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=90 is_in=0 shape=[1x1x1x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] gdma cmd_id bd_id=107 gdma_id=46 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=92 layer_type=Rope layer_name=
[bmprofile] tensor_id=88 is_in=1 shape=[1x1x8x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=19456 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=89 is_in=1 shape=[1x1x1x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=90 is_in=1 shape=[1x1x1x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=92 is_in=0 shape=[1x1x8x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=109 gdma_id=46 bd_func=3
[bmprofile] bd cmd_id bd_id=110 gdma_id=46 bd_func=3
[bmprofile] bd cmd_id bd_id=111 gdma_id=46 bd_func=3
[bmprofile] bd cmd_id bd_id=112 gdma_id=46 bd_func=3
[bmprofile] bd cmd_id bd_id=113 gdma_id=46 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=94 layer_type=Concat layer_name=
[bmprofile] tensor_id=92 is_in=1 shape=[1x1x8x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=91 is_in=1 shape=[1x1x8x192] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=94 is_in=0 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=8 l2addr=0
[bmprofile] bd cmd_id bd_id=114 gdma_id=46 bd_func=3
[bmprofile] bd cmd_id bd_id=115 gdma_id=46 bd_func=3
[bmprofile] local_layer: layer_id=93 layer_type=Load layer_name=
[bmprofile] tensor_id=75 is_in=1 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=4326944768 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=93 is_in=0 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2 l2addr=0
[bmprofile] gdma cmd_id bd_id=113 gdma_id=47 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=95 layer_type=Slice layer_name=
[bmprofile] tensor_id=93 is_in=1 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2 l2addr=0
[bmprofile] tensor_id=95 is_in=0 shape=[1x1x2x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=17152 nslice=1 hslice=2 l2addr=0
[bmprofile] bd cmd_id bd_id=116 gdma_id=47 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=97 layer_type=Slice layer_name=
[bmprofile] tensor_id=93 is_in=1 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2 l2addr=0
[bmprofile] tensor_id=97 is_in=0 shape=[1x1x2x192] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=2 l2addr=0
[bmprofile] bd cmd_id bd_id=117 gdma_id=47 bd_func=3
[bmprofile] local_layer: layer_id=98 layer_type=Rope layer_name=
[bmprofile] tensor_id=95 is_in=1 shape=[1x1x2x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=17152 nslice=1 hslice=2 l2addr=0
[bmprofile] tensor_id=89 is_in=1 shape=[1x1x1x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=65536 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=90 is_in=1 shape=[1x1x1x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=81920 nslice=1 hslice=1 l2addr=0
[bmprofile] tensor_id=98 is_in=0 shape=[1x1x2x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2 l2addr=0
[bmprofile] bd cmd_id bd_id=118 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=119 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=120 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=121 gdma_id=47 bd_func=3
[bmprofile] bd cmd_id bd_id=122 gdma_id=47 bd_func=3
[bmprofile] local_layer: layer_id=96 layer_type=Store layer_name=
[bmprofile] tensor_id=94 is_in=1 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=8 l2addr=0
[bmprofile] tensor_id=96 is_in=0 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=8 l2addr=0
[bmprofile] gdma cmd_id bd_id=116 gdma_id=48 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=99 layer_type=Concat layer_name=
[bmprofile] tensor_id=98 is_in=1 shape=[1x1x2x64] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2 l2addr=0
[bmprofile] tensor_id=97 is_in=1 shape=[1x1x2x192] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=2 l2addr=0
[bmprofile] tensor_id=99 is_in=0 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2 l2addr=0
[bmprofile] bd cmd_id bd_id=123 gdma_id=48 bd_func=3
[bmprofile] bd cmd_id bd_id=124 gdma_id=48 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=100 layer_type=Store layer_name=
[bmprofile] tensor_id=99 is_in=1 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2 l2addr=0
[bmprofile] tensor_id=100 is_in=0 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2 l2addr=0
[bmprofile] gdma cmd_id bd_id=124 gdma_id=49 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=103 layer_type=Concat layer_name=
[bmprofile] tensor_id=-4 is_in=1 shape=[1x2048x2x256] dtype=8 is_const=0 gaddr=4322717696 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=100 is_in=1 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=4324814848 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=103 is_in=0 shape=[1x2049x2x256] dtype=8 is_const=0 gaddr=4322717696 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=104 layer_type=Concat layer_name=
[bmprofile] tensor_id=-5 is_in=1 shape=[1x2048x2x256] dtype=8 is_const=0 gaddr=4324818944 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=66 is_in=1 shape=[1x1x2x256] dtype=8 is_const=0 gaddr=4326916096 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=104 is_in=0 shape=[1x2049x2x256] dtype=8 is_const=0 gaddr=4324818944 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0

[bmprofile] global_layer: layer_id=105 layer_type=FAttention layer_name=
[bmprofile] tensor_id=96 is_in=1 shape=[1x1x8x256] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=103 is_in=1 shape=[1x2049x2x256] dtype=8 is_const=0 gaddr=4322717696 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=104 is_in=1 shape=[1x2049x2x256] dtype=8 is_const=0 gaddr=4324818944 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=-6 is_in=1 shape=[1x1x1x2049] dtype=8 is_const=0 gaddr=4322709504 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=105 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] bd cmd_id bd_id=125 gdma_id=49 bd_func=5
[bmprofile] gdma cmd_id bd_id=125 gdma_id=50 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=126 gdma_id=50 bd_func=3
[bmprofile] bd cmd_id bd_id=127 gdma_id=50 bd_func=3
[bmprofile] bd cmd_id bd_id=128 gdma_id=50 bd_func=3
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=128 gdma_id=51 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=128 gdma_id=52 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=128 gdma_id=53 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=129 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=130 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=131 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=132 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=133 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=134 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=135 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=136 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=137 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=138 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=139 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=140 gdma_id=53 bd_func=1
[bmprofile] bd cmd_id bd_id=141 gdma_id=53 bd_func=1
[bmprofile] bd cmd_id bd_id=142 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=143 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=144 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=145 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=146 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=147 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=148 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=149 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=150 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=151 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=152 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=153 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=154 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=155 gdma_id=53 bd_func=9
[bmprofile] bd cmd_id bd_id=156 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=157 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=158 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=159 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=160 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=161 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=162 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=163 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=164 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=165 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=166 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=167 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=168 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=169 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=170 gdma_id=53 bd_func=9
[bmprofile] bd cmd_id bd_id=171 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=172 gdma_id=53 bd_func=1
[bmprofile] bd cmd_id bd_id=173 gdma_id=53 bd_func=1
[bmprofile] bd cmd_id bd_id=174 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=175 gdma_id=53 bd_func=3
[bmprofile] bd cmd_id bd_id=176 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=177 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=178 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=179 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=180 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=181 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=182 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=183 gdma_id=53 bd_func=2
[bmprofile] bd cmd_id bd_id=184 gdma_id=53 bd_func=3
[bmprofile] gdma cmd_id bd_id=128 gdma_id=54 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=128 gdma_id=55 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=128 gdma_id=56 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=185 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=186 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=187 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=188 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=189 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=190 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=191 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=192 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=193 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=194 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=195 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=196 gdma_id=56 bd_func=1
[bmprofile] bd cmd_id bd_id=197 gdma_id=56 bd_func=1
[bmprofile] bd cmd_id bd_id=198 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=199 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=200 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=201 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=202 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=203 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=204 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=205 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=206 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=207 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=208 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=209 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=210 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=211 gdma_id=56 bd_func=9
[bmprofile] bd cmd_id bd_id=212 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=213 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=214 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=215 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=216 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=217 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=218 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=219 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=220 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=221 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=222 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=223 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=224 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=225 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=226 gdma_id=56 bd_func=9
[bmprofile] bd cmd_id bd_id=227 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=228 gdma_id=56 bd_func=1
[bmprofile] bd cmd_id bd_id=229 gdma_id=56 bd_func=1
[bmprofile] bd cmd_id bd_id=230 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=231 gdma_id=56 bd_func=3
[bmprofile] bd cmd_id bd_id=232 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=233 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=234 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=235 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=236 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=237 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=238 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=239 gdma_id=56 bd_func=2
[bmprofile] bd cmd_id bd_id=240 gdma_id=56 bd_func=3
[bmprofile] gdma cmd_id bd_id=184 gdma_id=57 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=184 gdma_id=58 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=184 gdma_id=59 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=241 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=242 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=243 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=244 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=245 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=246 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=247 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=248 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=249 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=250 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=251 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=252 gdma_id=59 bd_func=1
[bmprofile] bd cmd_id bd_id=253 gdma_id=59 bd_func=1
[bmprofile] bd cmd_id bd_id=254 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=255 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=256 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=257 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=258 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=259 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=260 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=261 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=262 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=263 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=264 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=265 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=266 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=267 gdma_id=59 bd_func=9
[bmprofile] bd cmd_id bd_id=268 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=269 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=270 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=271 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=272 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=273 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=274 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=275 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=276 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=277 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=278 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=279 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=280 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=281 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=282 gdma_id=59 bd_func=9
[bmprofile] bd cmd_id bd_id=283 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=284 gdma_id=59 bd_func=1
[bmprofile] bd cmd_id bd_id=285 gdma_id=59 bd_func=1
[bmprofile] bd cmd_id bd_id=286 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=287 gdma_id=59 bd_func=3
[bmprofile] bd cmd_id bd_id=288 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=289 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=290 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=291 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=292 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=293 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=294 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=295 gdma_id=59 bd_func=2
[bmprofile] bd cmd_id bd_id=296 gdma_id=59 bd_func=3
[bmprofile] gdma cmd_id bd_id=240 gdma_id=60 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=240 gdma_id=61 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=240 gdma_id=62 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=297 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=298 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=299 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=300 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=301 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=302 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=303 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=304 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=305 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=306 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=307 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=308 gdma_id=62 bd_func=1
[bmprofile] bd cmd_id bd_id=309 gdma_id=62 bd_func=1
[bmprofile] bd cmd_id bd_id=310 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=311 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=312 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=313 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=314 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=315 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=316 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=317 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=318 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=319 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=320 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=321 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=322 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=323 gdma_id=62 bd_func=9
[bmprofile] bd cmd_id bd_id=324 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=325 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=326 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=327 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=328 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=329 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=330 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=331 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=332 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=333 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=334 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=335 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=336 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=337 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=338 gdma_id=62 bd_func=9
[bmprofile] bd cmd_id bd_id=339 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=340 gdma_id=62 bd_func=1
[bmprofile] bd cmd_id bd_id=341 gdma_id=62 bd_func=1
[bmprofile] bd cmd_id bd_id=342 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=343 gdma_id=62 bd_func=3
[bmprofile] bd cmd_id bd_id=344 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=345 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=346 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=347 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=348 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=349 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=350 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=351 gdma_id=62 bd_func=2
[bmprofile] bd cmd_id bd_id=352 gdma_id=62 bd_func=3
[bmprofile] gdma cmd_id bd_id=296 gdma_id=63 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=296 gdma_id=64 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=296 gdma_id=65 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=353 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=354 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=355 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=356 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=357 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=358 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=359 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=360 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=361 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=362 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=363 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=364 gdma_id=65 bd_func=1
[bmprofile] bd cmd_id bd_id=365 gdma_id=65 bd_func=1
[bmprofile] bd cmd_id bd_id=366 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=367 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=368 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=369 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=370 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=371 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=372 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=373 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=374 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=375 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=376 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=377 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=378 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=379 gdma_id=65 bd_func=9
[bmprofile] bd cmd_id bd_id=380 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=381 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=382 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=383 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=384 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=385 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=386 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=387 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=388 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=389 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=390 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=391 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=392 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=393 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=394 gdma_id=65 bd_func=9
[bmprofile] bd cmd_id bd_id=395 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=396 gdma_id=65 bd_func=1
[bmprofile] bd cmd_id bd_id=397 gdma_id=65 bd_func=1
[bmprofile] bd cmd_id bd_id=398 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=399 gdma_id=65 bd_func=3
[bmprofile] bd cmd_id bd_id=400 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=401 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=402 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=403 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=404 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=405 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=406 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=407 gdma_id=65 bd_func=2
[bmprofile] bd cmd_id bd_id=408 gdma_id=65 bd_func=3
[bmprofile] gdma cmd_id bd_id=352 gdma_id=66 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=352 gdma_id=67 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=352 gdma_id=68 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=409 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=410 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=411 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=412 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=413 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=414 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=415 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=416 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=417 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=418 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=419 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=420 gdma_id=68 bd_func=1
[bmprofile] bd cmd_id bd_id=421 gdma_id=68 bd_func=1
[bmprofile] bd cmd_id bd_id=422 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=423 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=424 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=425 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=426 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=427 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=428 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=429 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=430 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=431 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=432 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=433 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=434 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=435 gdma_id=68 bd_func=9
[bmprofile] bd cmd_id bd_id=436 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=437 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=438 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=439 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=440 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=441 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=442 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=443 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=444 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=445 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=446 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=447 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=448 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=449 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=450 gdma_id=68 bd_func=9
[bmprofile] bd cmd_id bd_id=451 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=452 gdma_id=68 bd_func=1
[bmprofile] bd cmd_id bd_id=453 gdma_id=68 bd_func=1
[bmprofile] bd cmd_id bd_id=454 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=455 gdma_id=68 bd_func=3
[bmprofile] bd cmd_id bd_id=456 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=457 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=458 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=459 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=460 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=461 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=462 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=463 gdma_id=68 bd_func=2
[bmprofile] bd cmd_id bd_id=464 gdma_id=68 bd_func=3
[bmprofile] gdma cmd_id bd_id=408 gdma_id=69 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=408 gdma_id=70 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=408 gdma_id=71 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=465 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=466 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=467 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=468 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=469 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=470 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=471 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=472 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=473 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=474 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=475 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=476 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=477 gdma_id=71 bd_func=1
[bmprofile] bd cmd_id bd_id=478 gdma_id=71 bd_func=1
[bmprofile] bd cmd_id bd_id=479 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=480 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=481 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=482 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=483 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=484 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=485 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=486 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=487 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=488 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=489 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=490 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=491 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=492 gdma_id=71 bd_func=9
[bmprofile] bd cmd_id bd_id=493 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=494 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=495 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=496 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=497 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=498 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=499 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=500 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=501 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=502 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=503 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=504 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=505 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=506 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=507 gdma_id=71 bd_func=9
[bmprofile] bd cmd_id bd_id=508 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=509 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=510 gdma_id=71 bd_func=1
[bmprofile] bd cmd_id bd_id=511 gdma_id=71 bd_func=1
[bmprofile] bd cmd_id bd_id=512 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=513 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=514 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=515 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=516 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=517 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=518 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=519 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=520 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=521 gdma_id=71 bd_func=2
[bmprofile] bd cmd_id bd_id=522 gdma_id=71 bd_func=3
[bmprofile] end parallel.
[bmprofile] bd cmd_id bd_id=523 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=524 gdma_id=71 bd_func=3
[bmprofile] bd cmd_id bd_id=525 gdma_id=71 bd_func=3
[bmprofile] gdma cmd_id bd_id=525 gdma_id=72 gdma_dir=0 gdma_func=0

[bmprofile] global_layer: layer_id=106 layer_type=Reshape layer_name=
[bmprofile] tensor_id=62 is_in=1 shape=[1x1x8x256] dtype=0 is_const=0 gaddr=4326928384 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=106 is_in=0 shape=[1x1x2048] dtype=0 is_const=0 gaddr=4326928384 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=108 layer_type=Load layer_name=
[bmprofile] tensor_id=106 is_in=1 shape=[1x1x2048] dtype=0 is_const=0 gaddr=4326928384 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=108 is_in=0 shape=[1x1x2048] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=525 gdma_id=73 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=110 layer_type=Cast layer_name=
[bmprofile] tensor_id=108 is_in=1 shape=[1x1x2048] dtype=0 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=110 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=2048 l2addr=0
[bmprofile] bd cmd_id bd_id=526 gdma_id=73 bd_func=3
[bmprofile] local_layer: layer_id=109 layer_type=Load layer_name=
[bmprofile] tensor_id=105 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=109 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=525 gdma_id=74 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=111 layer_type=Mul layer_name=
[bmprofile] tensor_id=109 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=110 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=111 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2048 l2addr=0
[bmprofile] bd cmd_id bd_id=527 gdma_id=74 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=112 layer_type=Store layer_name=
[bmprofile] tensor_id=111 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=112 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=527 gdma_id=75 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=115 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=112 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=37 is_in=1 shape=[2048x1024] dtype=3 is_const=1 gaddr=4315385856 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=36 is_in=1 shape=[64x32x16] dtype=8 is_const=1 gaddr=4315320320 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=35 is_in=1 shape=[64x32x16] dtype=3 is_const=1 gaddr=4315287552 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=115 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=527 gdma_id=76 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=527 gdma_id=77 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=527 gdma_id=78 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=527 gdma_id=79 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=527 gdma_id=80 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=528 gdma_id=79 bd_func=3
[bmprofile] bd cmd_id bd_id=529 gdma_id=79 bd_func=3
[bmprofile] bd cmd_id bd_id=530 gdma_id=79 bd_func=3
[bmprofile] bd cmd_id bd_id=531 gdma_id=79 bd_func=3
[bmprofile] bd cmd_id bd_id=532 gdma_id=79 bd_func=3
[bmprofile] bd cmd_id bd_id=533 gdma_id=79 bd_func=5
[bmprofile] bd cmd_id bd_id=534 gdma_id=79 bd_func=2
[bmprofile] bd cmd_id bd_id=535 gdma_id=79 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=535 gdma_id=81 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=536 gdma_id=80 bd_func=3
[bmprofile] bd cmd_id bd_id=537 gdma_id=80 bd_func=3
[bmprofile] bd cmd_id bd_id=538 gdma_id=80 bd_func=3
[bmprofile] bd cmd_id bd_id=539 gdma_id=80 bd_func=3
[bmprofile] bd cmd_id bd_id=540 gdma_id=80 bd_func=3
[bmprofile] bd cmd_id bd_id=541 gdma_id=80 bd_func=2
[bmprofile] bd cmd_id bd_id=542 gdma_id=80 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=542 gdma_id=82 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=117 layer_type=Load layer_name=
[bmprofile] tensor_id=-7 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4322701312 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=117 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=542 gdma_id=83 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=118 layer_type=Load layer_name=
[bmprofile] tensor_id=115 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=118 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=542 gdma_id=84 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=120 layer_type=Add layer_name=
[bmprofile] tensor_id=117 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=118 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=120 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] bd cmd_id bd_id=543 gdma_id=84 bd_func=3
[bmprofile] local_layer: layer_id=119 layer_type=Load layer_name=
[bmprofile] tensor_id=30 is_in=1 shape=[1x1x2048] dtype=8 is_const=1 gaddr=4314730496 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=119 is_in=0 shape=[1x1x2048] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=542 gdma_id=85 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=122 layer_type=RMSNorm layer_name=
[bmprofile] tensor_id=120 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=119 is_in=1 shape=[1x1x2048] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=122 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=2048 l2addr=0
[bmprofile] bd cmd_id bd_id=544 gdma_id=85 bd_func=3
[bmprofile] bd cmd_id bd_id=545 gdma_id=85 bd_func=3
[bmprofile] bd cmd_id bd_id=546 gdma_id=85 bd_func=1
[bmprofile] bd cmd_id bd_id=547 gdma_id=85 bd_func=1
[bmprofile] bd cmd_id bd_id=548 gdma_id=85 bd_func=3
[bmprofile] bd cmd_id bd_id=549 gdma_id=85 bd_func=9
[bmprofile] bd cmd_id bd_id=550 gdma_id=85 bd_func=13
[bmprofile] bd cmd_id bd_id=551 gdma_id=85 bd_func=3
[bmprofile] bd cmd_id bd_id=552 gdma_id=85 bd_func=3
[bmprofile] bd cmd_id bd_id=553 gdma_id=85 bd_func=3
[bmprofile] bd cmd_id bd_id=554 gdma_id=85 bd_func=5
[bmprofile] bd cmd_id bd_id=555 gdma_id=85 bd_func=3
[bmprofile] local_layer: layer_id=121 layer_type=Store layer_name=
[bmprofile] tensor_id=120 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=121 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=543 gdma_id=86 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=123 layer_type=Store layer_name=
[bmprofile] tensor_id=122 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=123 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=555 gdma_id=87 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=126 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=123 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326952960 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=26 is_in=1 shape=[6144x1024] dtype=3 is_const=1 gaddr=4301852672 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=25 is_in=1 shape=[64x96x16] dtype=8 is_const=1 gaddr=4301656064 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=24 is_in=1 shape=[64x96x16] dtype=3 is_const=1 gaddr=4301557760 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=126 is_in=0 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=555 gdma_id=88 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=555 gdma_id=89 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=555 gdma_id=90 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=555 gdma_id=91 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=555 gdma_id=92 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=556 gdma_id=91 bd_func=3
[bmprofile] bd cmd_id bd_id=557 gdma_id=91 bd_func=3
[bmprofile] bd cmd_id bd_id=558 gdma_id=91 bd_func=3
[bmprofile] bd cmd_id bd_id=559 gdma_id=91 bd_func=3
[bmprofile] bd cmd_id bd_id=560 gdma_id=91 bd_func=3
[bmprofile] bd cmd_id bd_id=561 gdma_id=91 bd_func=5
[bmprofile] bd cmd_id bd_id=562 gdma_id=91 bd_func=2
[bmprofile] bd cmd_id bd_id=563 gdma_id=91 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=563 gdma_id=93 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=563 gdma_id=94 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=564 gdma_id=92 bd_func=3
[bmprofile] bd cmd_id bd_id=565 gdma_id=92 bd_func=3
[bmprofile] bd cmd_id bd_id=566 gdma_id=92 bd_func=3
[bmprofile] bd cmd_id bd_id=567 gdma_id=92 bd_func=3
[bmprofile] bd cmd_id bd_id=568 gdma_id=92 bd_func=3
[bmprofile] bd cmd_id bd_id=569 gdma_id=92 bd_func=2
[bmprofile] bd cmd_id bd_id=570 gdma_id=92 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=570 gdma_id=95 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=570 gdma_id=96 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=571 gdma_id=94 bd_func=3
[bmprofile] bd cmd_id bd_id=572 gdma_id=94 bd_func=3
[bmprofile] bd cmd_id bd_id=573 gdma_id=94 bd_func=3
[bmprofile] bd cmd_id bd_id=574 gdma_id=94 bd_func=3
[bmprofile] bd cmd_id bd_id=575 gdma_id=94 bd_func=3
[bmprofile] bd cmd_id bd_id=576 gdma_id=94 bd_func=2
[bmprofile] bd cmd_id bd_id=577 gdma_id=94 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=577 gdma_id=97 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=577 gdma_id=98 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=578 gdma_id=96 bd_func=3
[bmprofile] bd cmd_id bd_id=579 gdma_id=96 bd_func=3
[bmprofile] bd cmd_id bd_id=580 gdma_id=96 bd_func=3
[bmprofile] bd cmd_id bd_id=581 gdma_id=96 bd_func=3
[bmprofile] bd cmd_id bd_id=582 gdma_id=96 bd_func=3
[bmprofile] bd cmd_id bd_id=583 gdma_id=96 bd_func=2
[bmprofile] bd cmd_id bd_id=584 gdma_id=96 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=584 gdma_id=99 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=584 gdma_id=100 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=585 gdma_id=98 bd_func=3
[bmprofile] bd cmd_id bd_id=586 gdma_id=98 bd_func=3
[bmprofile] bd cmd_id bd_id=587 gdma_id=98 bd_func=3
[bmprofile] bd cmd_id bd_id=588 gdma_id=98 bd_func=3
[bmprofile] bd cmd_id bd_id=589 gdma_id=98 bd_func=3
[bmprofile] bd cmd_id bd_id=590 gdma_id=98 bd_func=2
[bmprofile] bd cmd_id bd_id=591 gdma_id=98 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=591 gdma_id=101 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=592 gdma_id=100 bd_func=3
[bmprofile] bd cmd_id bd_id=593 gdma_id=100 bd_func=3
[bmprofile] bd cmd_id bd_id=594 gdma_id=100 bd_func=3
[bmprofile] bd cmd_id bd_id=595 gdma_id=100 bd_func=3
[bmprofile] bd cmd_id bd_id=596 gdma_id=100 bd_func=3
[bmprofile] bd cmd_id bd_id=597 gdma_id=100 bd_func=2
[bmprofile] bd cmd_id bd_id=598 gdma_id=100 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=598 gdma_id=102 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=127 layer_type=Cast layer_name=
[bmprofile] tensor_id=126 is_in=1 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=127 is_in=0 shape=[1x1x6144] dtype=0 is_const=0 gaddr=4326957056 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=598 gdma_id=103 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=599 gdma_id=103 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=599 gdma_id=104 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=128 layer_type=Active layer_name=
[bmprofile] tensor_id=127 is_in=1 shape=[1x1x6144] dtype=0 is_const=0 gaddr=4326957056 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=128 is_in=0 shape=[1x1x6144] dtype=0 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=599 gdma_id=105 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=600 gdma_id=105 bd_func=5
[bmprofile] bd cmd_id bd_id=601 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=602 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=603 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=604 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=605 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=606 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=607 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=608 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=609 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=610 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=611 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=612 gdma_id=105 bd_func=9
[bmprofile] bd cmd_id bd_id=613 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=614 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=615 gdma_id=105 bd_func=3
[bmprofile] bd cmd_id bd_id=616 gdma_id=105 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=616 gdma_id=106 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=129 layer_type=Cast layer_name=
[bmprofile] tensor_id=128 is_in=1 shape=[1x1x6144] dtype=0 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=129 is_in=0 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326957056 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=616 gdma_id=107 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=617 gdma_id=107 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=617 gdma_id=108 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=130 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=123 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326952960 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=29 is_in=1 shape=[6144x1024] dtype=3 is_const=1 gaddr=4308439040 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=28 is_in=1 shape=[64x96x16] dtype=8 is_const=1 gaddr=4308242432 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=27 is_in=1 shape=[64x96x16] dtype=3 is_const=1 gaddr=4308144128 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=130 is_in=0 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=617 gdma_id=109 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=617 gdma_id=110 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=617 gdma_id=111 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=617 gdma_id=112 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=617 gdma_id=113 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=618 gdma_id=112 bd_func=3
[bmprofile] bd cmd_id bd_id=619 gdma_id=112 bd_func=3
[bmprofile] bd cmd_id bd_id=620 gdma_id=112 bd_func=3
[bmprofile] bd cmd_id bd_id=621 gdma_id=112 bd_func=3
[bmprofile] bd cmd_id bd_id=622 gdma_id=112 bd_func=3
[bmprofile] bd cmd_id bd_id=623 gdma_id=112 bd_func=5
[bmprofile] bd cmd_id bd_id=624 gdma_id=112 bd_func=2
[bmprofile] bd cmd_id bd_id=625 gdma_id=112 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=625 gdma_id=114 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=625 gdma_id=115 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=626 gdma_id=113 bd_func=3
[bmprofile] bd cmd_id bd_id=627 gdma_id=113 bd_func=3
[bmprofile] bd cmd_id bd_id=628 gdma_id=113 bd_func=3
[bmprofile] bd cmd_id bd_id=629 gdma_id=113 bd_func=3
[bmprofile] bd cmd_id bd_id=630 gdma_id=113 bd_func=3
[bmprofile] bd cmd_id bd_id=631 gdma_id=113 bd_func=2
[bmprofile] bd cmd_id bd_id=632 gdma_id=113 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=632 gdma_id=116 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=632 gdma_id=117 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=633 gdma_id=115 bd_func=3
[bmprofile] bd cmd_id bd_id=634 gdma_id=115 bd_func=3
[bmprofile] bd cmd_id bd_id=635 gdma_id=115 bd_func=3
[bmprofile] bd cmd_id bd_id=636 gdma_id=115 bd_func=3
[bmprofile] bd cmd_id bd_id=637 gdma_id=115 bd_func=3
[bmprofile] bd cmd_id bd_id=638 gdma_id=115 bd_func=2
[bmprofile] bd cmd_id bd_id=639 gdma_id=115 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=639 gdma_id=118 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=639 gdma_id=119 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=640 gdma_id=117 bd_func=3
[bmprofile] bd cmd_id bd_id=641 gdma_id=117 bd_func=3
[bmprofile] bd cmd_id bd_id=642 gdma_id=117 bd_func=3
[bmprofile] bd cmd_id bd_id=643 gdma_id=117 bd_func=3
[bmprofile] bd cmd_id bd_id=644 gdma_id=117 bd_func=3
[bmprofile] bd cmd_id bd_id=645 gdma_id=117 bd_func=2
[bmprofile] bd cmd_id bd_id=646 gdma_id=117 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=646 gdma_id=120 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=646 gdma_id=121 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=647 gdma_id=119 bd_func=3
[bmprofile] bd cmd_id bd_id=648 gdma_id=119 bd_func=3
[bmprofile] bd cmd_id bd_id=649 gdma_id=119 bd_func=3
[bmprofile] bd cmd_id bd_id=650 gdma_id=119 bd_func=3
[bmprofile] bd cmd_id bd_id=651 gdma_id=119 bd_func=3
[bmprofile] bd cmd_id bd_id=652 gdma_id=119 bd_func=2
[bmprofile] bd cmd_id bd_id=653 gdma_id=119 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=653 gdma_id=122 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=654 gdma_id=121 bd_func=3
[bmprofile] bd cmd_id bd_id=655 gdma_id=121 bd_func=3
[bmprofile] bd cmd_id bd_id=656 gdma_id=121 bd_func=3
[bmprofile] bd cmd_id bd_id=657 gdma_id=121 bd_func=3
[bmprofile] bd cmd_id bd_id=658 gdma_id=121 bd_func=3
[bmprofile] bd cmd_id bd_id=659 gdma_id=121 bd_func=2
[bmprofile] bd cmd_id bd_id=660 gdma_id=121 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=660 gdma_id=123 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=131 layer_type=Mul layer_name=
[bmprofile] tensor_id=129 is_in=1 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326957056 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=130 is_in=1 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=131 is_in=0 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=660 gdma_id=124 gdma_dir=0 gdma_func=1
[bmprofile] gdma cmd_id bd_id=660 gdma_id=125 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] bd cmd_id bd_id=661 gdma_id=125 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=661 gdma_id=126 gdma_dir=0 gdma_func=1
[bmprofile] end parallel.

[bmprofile] global_layer: layer_id=132 layer_type=A16MatMul layer_name=
[bmprofile] tensor_id=131 is_in=1 shape=[1x1x6144] dtype=8 is_const=0 gaddr=4326936576 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=23 is_in=1 shape=[2048x3072] dtype=3 is_const=1 gaddr=4295266304 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=22 is_in=1 shape=[64x32x48] dtype=8 is_const=1 gaddr=4295069696 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=21 is_in=1 shape=[64x32x48] dtype=3 is_const=1 gaddr=4294971392 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=132 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=661 gdma_id=127 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=661 gdma_id=128 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=661 gdma_id=129 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=661 gdma_id=130 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=661 gdma_id=131 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=662 gdma_id=130 bd_func=3
[bmprofile] bd cmd_id bd_id=663 gdma_id=130 bd_func=3
[bmprofile] bd cmd_id bd_id=664 gdma_id=130 bd_func=3
[bmprofile] bd cmd_id bd_id=665 gdma_id=130 bd_func=3
[bmprofile] bd cmd_id bd_id=666 gdma_id=130 bd_func=3
[bmprofile] bd cmd_id bd_id=667 gdma_id=130 bd_func=5
[bmprofile] bd cmd_id bd_id=668 gdma_id=130 bd_func=2
[bmprofile] bd cmd_id bd_id=669 gdma_id=130 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=669 gdma_id=132 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=669 gdma_id=133 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=670 gdma_id=131 bd_func=3
[bmprofile] bd cmd_id bd_id=671 gdma_id=131 bd_func=3
[bmprofile] bd cmd_id bd_id=672 gdma_id=131 bd_func=3
[bmprofile] bd cmd_id bd_id=673 gdma_id=131 bd_func=3
[bmprofile] bd cmd_id bd_id=674 gdma_id=131 bd_func=3
[bmprofile] bd cmd_id bd_id=675 gdma_id=131 bd_func=2
[bmprofile] bd cmd_id bd_id=676 gdma_id=131 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=676 gdma_id=134 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=676 gdma_id=135 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=677 gdma_id=133 bd_func=3
[bmprofile] bd cmd_id bd_id=678 gdma_id=133 bd_func=3
[bmprofile] bd cmd_id bd_id=679 gdma_id=133 bd_func=3
[bmprofile] bd cmd_id bd_id=680 gdma_id=133 bd_func=3
[bmprofile] bd cmd_id bd_id=681 gdma_id=133 bd_func=3
[bmprofile] bd cmd_id bd_id=682 gdma_id=133 bd_func=2
[bmprofile] bd cmd_id bd_id=683 gdma_id=133 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=683 gdma_id=136 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=683 gdma_id=137 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=684 gdma_id=135 bd_func=3
[bmprofile] bd cmd_id bd_id=685 gdma_id=135 bd_func=3
[bmprofile] bd cmd_id bd_id=686 gdma_id=135 bd_func=3
[bmprofile] bd cmd_id bd_id=687 gdma_id=135 bd_func=3
[bmprofile] bd cmd_id bd_id=688 gdma_id=135 bd_func=3
[bmprofile] bd cmd_id bd_id=689 gdma_id=135 bd_func=2
[bmprofile] bd cmd_id bd_id=690 gdma_id=135 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=690 gdma_id=138 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=690 gdma_id=139 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=691 gdma_id=137 bd_func=3
[bmprofile] bd cmd_id bd_id=692 gdma_id=137 bd_func=3
[bmprofile] bd cmd_id bd_id=693 gdma_id=137 bd_func=3
[bmprofile] bd cmd_id bd_id=694 gdma_id=137 bd_func=3
[bmprofile] bd cmd_id bd_id=695 gdma_id=137 bd_func=3
[bmprofile] bd cmd_id bd_id=696 gdma_id=137 bd_func=2
[bmprofile] bd cmd_id bd_id=697 gdma_id=137 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=697 gdma_id=140 gdma_dir=0 gdma_func=0
[bmprofile] gdma cmd_id bd_id=697 gdma_id=141 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=698 gdma_id=139 bd_func=3
[bmprofile] bd cmd_id bd_id=699 gdma_id=139 bd_func=3
[bmprofile] bd cmd_id bd_id=700 gdma_id=139 bd_func=3
[bmprofile] bd cmd_id bd_id=701 gdma_id=139 bd_func=3
[bmprofile] bd cmd_id bd_id=702 gdma_id=139 bd_func=3
[bmprofile] bd cmd_id bd_id=703 gdma_id=139 bd_func=2
[bmprofile] bd cmd_id bd_id=704 gdma_id=139 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=704 gdma_id=142 gdma_dir=0 gdma_func=0
[bmprofile] bd cmd_id bd_id=705 gdma_id=141 bd_func=3
[bmprofile] bd cmd_id bd_id=706 gdma_id=141 bd_func=3
[bmprofile] bd cmd_id bd_id=707 gdma_id=141 bd_func=3
[bmprofile] bd cmd_id bd_id=708 gdma_id=141 bd_func=3
[bmprofile] bd cmd_id bd_id=709 gdma_id=141 bd_func=3
[bmprofile] bd cmd_id bd_id=710 gdma_id=141 bd_func=2
[bmprofile] bd cmd_id bd_id=711 gdma_id=141 bd_func=5
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] gdma cmd_id bd_id=711 gdma_id=143 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=134 layer_type=Load layer_name=
[bmprofile] tensor_id=121 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326948864 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=134 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=711 gdma_id=144 gdma_dir=0 gdma_func=0
[bmprofile] local_layer: layer_id=135 layer_type=Load layer_name=
[bmprofile] tensor_id=132 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=4326924288 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=135 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=711 gdma_id=145 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=137 layer_type=Add layer_name=
[bmprofile] tensor_id=134 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=32768 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=135 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=49152 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=137 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] bd cmd_id bd_id=712 gdma_id=145 bd_func=3
[bmprofile] local_layer: layer_id=136 layer_type=Load layer_name=
[bmprofile] tensor_id=45 is_in=1 shape=[1x1x2048] dtype=8 is_const=1 gaddr=4322426880 gsize=0 loffset=0 nslice=0 hslice=0 l2addr=0
[bmprofile] tensor_id=136 is_in=0 shape=[1x1x2048] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=711 gdma_id=146 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=138 layer_type=RMSNorm layer_name=
[bmprofile] tensor_id=137 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=0 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=136 is_in=1 shape=[1x1x2048] dtype=8 is_const=1 gaddr=0 gsize=0 loffset=16384 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=138 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=2048 l2addr=0
[bmprofile] bd cmd_id bd_id=713 gdma_id=146 bd_func=3
[bmprofile] bd cmd_id bd_id=714 gdma_id=146 bd_func=3
[bmprofile] bd cmd_id bd_id=715 gdma_id=146 bd_func=1
[bmprofile] bd cmd_id bd_id=716 gdma_id=146 bd_func=1
[bmprofile] bd cmd_id bd_id=717 gdma_id=146 bd_func=3
[bmprofile] bd cmd_id bd_id=718 gdma_id=146 bd_func=9
[bmprofile] bd cmd_id bd_id=719 gdma_id=146 bd_func=13
[bmprofile] bd cmd_id bd_id=720 gdma_id=146 bd_func=3
[bmprofile] bd cmd_id bd_id=721 gdma_id=146 bd_func=3
[bmprofile] bd cmd_id bd_id=722 gdma_id=146 bd_func=3
[bmprofile] bd cmd_id bd_id=723 gdma_id=146 bd_func=5
[bmprofile] bd cmd_id bd_id=724 gdma_id=146 bd_func=3
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] local_layer: layer_id=139 layer_type=Store layer_name=
[bmprofile] tensor_id=138 is_in=1 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=2048 l2addr=0
[bmprofile] tensor_id=139 is_in=0 shape=[1x1x2048] dtype=8 is_const=0 gaddr=0 gsize=0 loffset=98304 nslice=1 hslice=2048 l2addr=0
[bmprofile] gdma cmd_id bd_id=724 gdma_id=147 gdma_dir=0 gdma_func=0
[bmprofile] end parallel.
[bmprofile] start parallel.
[bmprofile] end parallel.
[bmprofile] insert bd end (cmd_id bd_id=725)
[bmprofile] bd cmd_id bd_id=725 gdma_id=0 bd_func=15
[bmprofile] insert gdma end (cmd_id gdma_id=148)
[bmprofile] gdma cmd_id bd_id=0 gdma_id=148 gdma_dir=0 gdma_func=6
[bmprofile] end to run subnet_id=0
